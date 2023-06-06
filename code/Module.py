import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st

def gkern(kernlen=16, nsig=3):
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)

class HA(nn.Module):
    # holistic attention module
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15).contiguous()
        soft_attention = min_max_norm(soft_attention).contiguous()
        x = torch.mul(x, soft_attention.max(attention)).contiguous()
        return x


class HA_returnTwo(nn.Module):
    # holistic attention module
    def __init__(self):
        super(HA_returnTwo, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15).contiguous()
        soft_attention = min_max_norm(soft_attention).contiguous()
        x = torch.mul(x, soft_attention.max(attention)).contiguous()
        return x, soft_attention
    

class channel_mask(nn.Module):
    def __init__(self, in_dim, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_dim // ratio, in_dim, 1, bias=False) 
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):  
        input = x
        avg_out = self.shared_MLP(self.avg_pool(input))
        cha_w = self.sigmoid(avg_out)
        # max_out = self.shared_MLP(self.max_pool(input))
        # cha_w = self.sigmoid(max_out) 
        return cha_w  


class spatial_mask(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        input = x
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        spa_w = self.sigmoid(self.conv(max_out))
        return spa_w    


class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_dim, out_dim, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_dim, out_dim, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_dim, out_dim, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_dim, out_dim, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_dim, out_dim, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_dim * 5, out_dim, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class ConvGRUCell(nn.Module):
    """
        ICLR2016: Delving Deeper into Convolutional Networks for Learning Video Representations
        url: https://arxiv.org/abs/1511.06432
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, cuda_flag=True):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.cuda_flag = cuda_flag
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        padding = self.kernel_size // 2
        self.reset_gate  = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=padding)
        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=padding)
        self.output_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=padding)
        # init
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden):
        if hidden is None:
           size_h    = [x.data.size()[0], self.hidden_channels] + list(x.data.size()[2:])
           if self.cuda_flag:
              hidden = torch.zeros(size_h).cuda()
           else:
              hidden = torch.zeros(size_h)

        inputs       = torch.cat((x, hidden), dim=1)
        reset_gate   = torch.sigmoid(self.reset_gate(inputs))
        update_gate  = torch.sigmoid(self.update_gate(inputs))

        reset_hidden = reset_gate * hidden
        reset_inputs = torch.tanh(self.output_gate(torch.cat((x, reset_hidden), dim=1)))
        new_hidden   = (1 - update_gate)*reset_inputs + update_gate*hidden

        return new_hidden


class BConv3(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, padding):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        input = x
        out = self.relu(self.bn1(self.conv1(input)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
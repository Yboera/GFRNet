import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention_One(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CoordAttention_One, self).__init__()
        self.pool_w_d, self.pool_h_d = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_f, self.pool_h_f = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv_h = nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.conv_cat = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)        

    def forward(self, focal, depth):
        short = focal
        n, c, H, W = focal.shape
        focal_h, focal_w = self.pool_h_f(focal), self.pool_w_f(focal).permute(0, 1, 3, 2)

        depth_h, depth_w = self.pool_h_d(depth), self.pool_w_d(depth).permute(0, 1, 3, 2)
        
        h_cat = self.conv_h(torch.cat([focal_h, depth_h], dim=1)) 
        w_cat = self.conv_w(torch.cat([focal_w, depth_w], dim=1))
    
        ful = torch.cat([h_cat, w_cat], dim=2)
        out = self.act1(self.bn1(self.conv_cat(ful)))

        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))

        return short * out_w * out_h + short 

class CoordAttention_Two(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CoordAttention_Two, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool_w_d, self.pool_h_d = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_f, self.pool_h_f = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        
        self.conv_h = nn.Sequential(nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0), self.relu,
                                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
        self.conv_w = nn.Sequential(nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0), self.relu,
                                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
            
        temp_c = max(8, in_channels // reduction)
        self.conv_cat = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
       
        # self.conv_h = nn.Sequential(nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0), self.relu)
        # self.conv_w = nn.Sequential(nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0), self.relu)
        
        # self.conv_cat = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(in_channels)
        # self.act1 = self.relu

        # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, focal, depth):
        short = focal
        n, c, H, W = focal.shape
        
        focal_h = self.pool_h_f(focal)
        focal_w = self.pool_w_f(focal).permute(0, 1, 3, 2)
        depth_h = self.pool_h_d(depth)
        depth_w = self.pool_w_d(depth).permute(0, 1, 3, 2)
        
        depth_h = depth_h.repeat(12, 1, 1, 1)
        depth_w = depth_w.repeat(12, 1, 1, 1)
        
        h_cat = self.conv_h(torch.cat([focal_h, depth_h], dim=1)) 
        w_cat = self.conv_w(torch.cat([focal_w, depth_w], dim=1))

        # h_cat = self.conv_h(focal_h * depth_h) 
        # w_cat = self.conv_w(focal_w * depth_w)

        ful = torch.cat([h_cat, w_cat], dim=2)
        out = self.act1(self.bn1(self.conv_cat(ful)))

        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        
        return short * out_w * out_h + short
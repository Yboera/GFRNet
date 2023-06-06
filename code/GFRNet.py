import torch
import torch.nn as nn
from torch.nn import functional as F
from Module import RFB, BConv3, channel_mask
from VGG import VGGNet
from CoordAttention import CoordAttention_Two

out_k = 64

class Compression(nn.Module):   
    def __init__(self, layer, mode):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        assert layer in [0, 1, 2, 3]
        self.layer = layer
        assert mode in ['f', 'r']
        self.mode = mode
        
        if layer == 0:
            if mode == 'f': com = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu, 
                                                nn.Conv2d(128, out_k, 3, 1, 1), self.relu)
            else:           com = nn.Sequential(RFB(128, out_k))
                                               
        elif layer == 1:
            if mode == 'f': com = nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                                nn.Conv2d(256, out_k, 3, 1, 1), self.relu)
            else:           com = nn.Sequential(RFB(256, out_k))
                                               
        elif layer == 2:
            if mode == 'f': com = nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu, 
                                                nn.Conv2d(512, out_k, 3, 1, 1), self.relu)
            else:           com = nn.Sequential(RFB(512, out_k)) 
                                               
        elif layer == 3:
            if mode == 'f': com = nn.Sequential(nn.Conv2d(512, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2), self.relu, 
                                                nn.Conv2d(512, out_k, 3, 1, 1), self.relu)
            else:           com = nn.Sequential(RFB(512, out_k))
                                               
        self.compression = com

    def forward(self, x): 
        return self.compression(x)


class FS_fuse(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_sum = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        self.conv_max = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        self.conv_out = nn.Sequential(BConv3(in_dim*2, in_dim, kernel_size=5, padding=2))

    def forward(self, fs):
        # ful_1 = self.conv_mul(fs)
        # ful_list = torch.chunk(ful_1, 12, dim=0)
        # ful_mul = ful_list[0]
        # for i in range(1, 12):
        #     ful_mul = ful_mul * ful_list[i]

        ful_1 = self.conv_sum(fs)
        ful_sum = ful_1.sum(axis=0).unsqueeze(0)
        
        ful_2 = self.conv_max(fs)
        ful_max = ful_2.max(dim=0)[0]
        ful_max = ful_max.unsqueeze(0)
        ful_max = ful_max * 12.0

        ful = torch.cat((ful_sum, ful_max), dim=1)
        res = self.conv_out(ful)
        return res


class ARM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tran_rgb = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        self.tran_fs = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        # self.compress = nn.Sequential(nn.Conv2d(in_dim, in_dim//4, kernel_size=1, padding=0), 
        #                               nn.Conv2d(in_dim//4, 1, kernel_size=1, padding=0))
        # self.tran = nn.Sequential(nn.Conv2d(in_dim, in_dim//4, kernel_size=3, padding=1), nn.BatchNorm2d(in_dim//4), self.relu,
        #                           nn.Conv2d(in_dim//4, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.get_gate = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0), self.relu,
                                      nn.Conv2d(in_dim, 1, kernel_size=1, padding=0))
        self.get_mask = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1), nn.BatchNorm2d(in_dim), self.relu,
                                  nn.Conv2d(in_dim, 1, kernel_size=1, padding=0), nn.Sigmoid())
        self.fuse = FS_fuse(in_dim)
        
    def forward(self, fs, rgb):
        rgb_g = self.tran_rgb(rgb)
        rgb_g = rgb_g.repeat(12, 1, 1, 1)        
        fs_g = self.tran_fs(fs)
        
        awm = F.adaptive_avg_pool2d(rgb_g * fs_g, 1) / (F.adaptive_avg_pool2d(rgb_g + fs_g, 1))
        gate = self.get_gate(awm)
        gate = F.softmax(gate, dim=0)
        fs = fs * gate
        
        mask= self.get_mask(fs * rgb)
        #he_w = nn.Sigmoid()(he)
        fs = fs * mask + fs
        
        focal_r = self.fuse(fs)

        return focal_r 


class DRM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.co_att = CoordAttention_Two(in_dim, in_dim)
        self.fuse = FS_fuse(in_dim)

    def forward(self, fs, dep):
        # fs_tmp = torch.cat(torch.chunk(fs, 12, dim=0), dim=1)
        # fs_tmp = self.cha(fs_tmp) * fs_tmp
        # fs_res = torch.cat(torch.chunk(fs_tmp, 12, dim=1), dim=0)
        # focal_d = self.fuse(fs_res)
        # focal = fs + dep
        focal = self.co_att(fs, dep)
        focal_d = self.fuse(focal)
    
        return focal_d

class GRFM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_fs_r = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        self.conv_fs_d = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        self.cha_r = channel_mask(in_dim)
        self.cha_d = channel_mask(in_dim)
        self.att_r = ARM(in_dim) 
        self.att_d = DRM(in_dim)
        self.fuse_focal = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        self.fuse_rd = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_dim), self.relu)        
        self.fuse = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_dim), self.relu)
        
    def forward(self, fs, rgb, dep):
        fs_r = self.conv_fs_r(fs)
        fs_d = self.conv_fs_d(fs)

        rgb = rgb + rgb * self.cha_r(rgb) 
        dep = dep + dep * self.cha_d(dep) 
        # rgb = rgb * self.cha_r(rgb)
        # dep = dep * self.cha_d(dep)
         
        focal_r = self.att_r(fs_r, rgb)
        focal_d = self.att_d(fs_d, dep)

        res_f = self.fuse_focal(torch.cat((focal_r * focal_d, focal_r + focal_d), dim=1))
        res_rd = self.fuse_rd(torch.cat((rgb, dep), dim=1))
        # data_rd = self.fuse_rd(torch.cat((rgb * dep, rgb + dep), dim=1))
        res = self.fuse(torch.cat((res_f, res_rd), dim=1))
        
        return res, rgb, dep

class Decoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_0 = BConv3(2 * in_dim, in_dim, 5, 2)
        self.conv_1 = BConv3(2 * in_dim, in_dim, 5, 2)
        self.conv_2 = BConv3(2 * in_dim, in_dim, 5, 2)
        self.conv_3 = BConv3(in_dim, in_dim, 5, 2)
        #self.conv_3 = nn.Sequential(nn.Conv2d(in_dim, in_dim, 5, 1, 2), nn.BatchNorm2d(in_dim), self.relu)
    
    def forward(self, list):
        layer=[]
        tmp = self.conv_3(list[3])

        layer.append(tmp)
        tmp = F.interpolate(tmp, scale_factor=2, mode='bilinear', align_corners=True)
        tmp = self.conv_2(torch.cat((tmp, list[2]),dim=1))
        layer.append(tmp)
        tmp = F.interpolate(tmp, scale_factor=2, mode='bilinear', align_corners=True)
        tmp = self.conv_1(torch.cat((tmp, list[1]),dim=1))
        layer.append(tmp)
        tmp = F.interpolate(tmp, scale_factor=2, mode='bilinear', align_corners=True)
        tmp = self.conv_0(torch.cat((tmp, list[0]),dim=1))
        layer.append(tmp)
        
        layer.reverse()
        return layer

class GFRNet(nn.Module):
    def __init__(self, backbone_fs, backbone_rgb, backbone_dep):
        super(GFRNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.backbone_fs  = backbone_fs
        self.backbone_rgb = backbone_rgb
        self.backbone_dep  = backbone_dep

        cp_f = []
        cp_f.append(Compression(layer=0, mode='f'))
        cp_f.append(Compression(layer=1, mode='f'))
        cp_f.append(Compression(layer=2, mode='f'))
        cp_f.append(Compression(layer=3, mode='f'))
        self.cp_fs = nn.ModuleList(cp_f)
        
        cp_r = []
        cp_r.append(Compression(layer=0, mode='f'))
        cp_r.append(Compression(layer=1, mode='f'))
        cp_r.append(Compression(layer=2, mode='f'))
        cp_r.append(Compression(layer=3, mode='f'))
        self.cp_rgb = nn.ModuleList(cp_r)
        
        cp_d = []
        cp_d.append(Compression(layer=0, mode='f'))
        cp_d.append(Compression(layer=1, mode='f'))
        cp_d.append(Compression(layer=2, mode='f'))
        cp_d.append(Compression(layer=3, mode='f'))
        self.cp_dep = nn.ModuleList(cp_d)

        l = []
        l.append(GRFM(out_k))
        l.append(GRFM(out_k))
        l.append(GRFM(out_k))
        l.append(GRFM(out_k))
        self.tri_fuse = nn.ModuleList(l)

        self.decoder = Decoder(out_k)

        self.r_coarse = nn.Conv2d(out_k, 1, 1, 1)
        self.d_coarse = nn.Conv2d(out_k, 1, 1, 1)
        #self.f_coarse = nn.Conv2d(out_k, 1, 1, 1)
                                
        co = []
        co.append(nn.Conv2d(out_k, 1, 1, 1))
        co.append(nn.Conv2d(out_k, 1, 1, 1))
        co.append(nn.Conv2d(out_k, 1, 1, 1))
        co.append(nn.Conv2d(out_k, 1, 1, 1))
        self.coarse = nn.ModuleList(co)

    def forward(self, fs, rgb, dep):
        fs_list, low_fs = self.backbone_fs(fs)              
        rgb_list, low_rgb = self.backbone_rgb(rgb)
        dep_list, low_dep = self.backbone_dep(dep)
        
        for i in range(4):
            fs_list[i] = self.cp_fs[i](fs_list[i])
            rgb_list[i] = self.cp_rgb[i](rgb_list[i])
            dep_list[i] = self.cp_dep[i](dep_list[i])

        en_list = []
        for i in range(4):          
            temp, co_r, co_d = self.tri_fuse[i](fs_list[i], rgb_list[i], dep_list[i]) 
            en_list.append(temp)
        
        co_r = self.r_coarse(co_r)
        co_d = self.d_coarse(co_d)
        
        layer = self.decoder(en_list)
        
        # low_r = self.maxpool(self.tran_r(low_rgb))
        # low_d = self.maxpool(self.tran_d(low_dep))
        # res[3] = res[3] + low_r + low_d
        # res[3] = self.refine(res[3])             

        for i in range(4):
            layer[i] = F.interpolate(layer[i], (256, 256), mode='bilinear', align_corners=True)
            layer[i] = self.coarse[i](layer[i]) 
           
        return layer, co_r, co_d
        
def build_net():
    model_fs = VGGNet(requires_grad=True)
    model_rgb = VGGNet(requires_grad=True)
    model_dep = VGGNet(requires_grad=True)
    return GFRNet(model_fs, model_rgb, model_dep)
 
 
import torch
from torch import nn
from torch.nn import functional as F
from modules import Inception

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False, se=True):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        self.se=se
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.selayer = SELayer(in_channels)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        """
        if self.se:
            y = self.selayer(y)
        """
        return y

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid):
        super(Encoder,self).__init__()
        self.first = BasicConv2d(C_in, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=False)
        self.enc1 = nn.Sequential(
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.enc2 = nn.Sequential(
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.down1 = BasicConv2d(C_hid, C_hid*2, kernel_size=3, stride=2,
                                padding=1, transpose=False, act_norm=True, se=False)
        self.enc3 = nn.Sequential(
            BasicConv2d(C_hid*2, C_hid*2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid*2, C_hid*2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.enc4 = nn.Sequential(
            BasicConv2d(C_hid*2, C_hid*2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid*2, C_hid*2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.down2 = BasicConv2d(C_hid*2, C_hid*4, kernel_size=3, stride=2,
                                padding=1, transpose=False, act_norm=True, se=False)
        self.enc5 = nn.Sequential(
            BasicConv2d(C_hid*4, C_hid*4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid*4, C_hid*4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.enc6 = nn.Sequential(
            BasicConv2d(C_hid*4, C_hid*4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid*4, C_hid*4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.down3 = BasicConv2d(C_hid*4, C_hid*8, kernel_size=3, stride=2,
                                padding=1, transpose=False, act_norm=True, se=False)
        self.enc7 = nn.Sequential(
            BasicConv2d(C_hid*8, C_hid*8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid*8, C_hid*8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.enc8 = nn.Sequential(
            BasicConv2d(C_hid*8, C_hid*8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True),
            BasicConv2d(C_hid*8, C_hid*8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True)
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        first = self.first(x)
        res1 = self.enc1(first) * 0.9 + first * 0.1
        res2 = self.enc2(res1) * 0.9 + res1 * 0.1
        res3 = self.down1(res2)
        res4 = self.enc3(res3) * 0.9 + res3 * 0.1
        res5 = self.enc4(res4) * 0.9 + res4 * 0.1
        res6 = self.down2(res5)
        res7 = self.enc5(res6) * 0.9 + res6 * 0.1
        #res8 = self.enc6(res7) * 0.9 + res7 * 0.1
        #res9 = self.down3(res8)
        #res10 = self.enc7(res9) * 0.9 + res9 * 0.1
        #res11 = self.enc8(res10) * 0.9 + res10 * 0.1
        return res7


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out):
        super(Decoder,self).__init__()
        self.dec1 = nn.Sequential(
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.dec2 = nn.Sequential(
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid, C_hid, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.up1 = BasicConv2d(C_hid, C_hid//2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=False)
        self.dec3 = nn.Sequential(
            BasicConv2d(C_hid//2, C_hid//2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid//2, C_hid//2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.dec4 = nn.Sequential(
            BasicConv2d(C_hid//2, C_hid//2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid//2, C_hid//2, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.up2 = BasicConv2d(C_hid//2, C_hid//4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=False)
        self.dec5 = nn.Sequential(
            BasicConv2d(C_hid//4, C_hid//4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid//4, C_hid//4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.dec6 = nn.Sequential(
            BasicConv2d(C_hid//4, C_hid//4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid//4, C_hid//4, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.up3 = BasicConv2d(C_hid//4, C_hid//8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=False)
        self.dec7 = nn.Sequential(
            BasicConv2d(C_hid//8, C_hid//8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True),
            BasicConv2d(C_hid//8, C_hid//8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True, se=True)
        )
        self.dec8 = nn.Sequential(
            BasicConv2d(C_hid//8, C_hid//8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True),
            BasicConv2d(C_hid//8, C_hid//8, kernel_size=3, stride=1,
                                padding=1, transpose=False, act_norm=True)
        )
        self.readout = nn.Conv2d(C_hid//4, C_out, 1)
    
    def forward(self, hid):
        res1 = self.dec1(hid) * 0.9 + hid * 0.1
        res2 = self.dec2(res1) * 0.9 + res1 * 0.1
        res3 = self.up1(F.interpolate(res2, scale_factor=2, mode='nearest'))
        res4 = self.dec3(res3) * 0.9 + res3 * 0.1
        res5 = self.dec4(res4) * 0.9 + res4 * 0.1
        res6 = self.up2(F.interpolate(res5, scale_factor=2, mode='nearest'))
        #res7 = self.dec5(res6) * 0.9 + res6 * 0.1
        #res8 = self.dec6(res7) * 0.9 + res7 * 0.1
        #res9 = self.up3(F.interpolate(res8, scale_factor=2, mode='nearest'))
        #res10 = self.dec7(res9) * 0.9 + res9 * 0.1
        #res11 = self.dec8(res10) * 0.9 + res10 * 0.1
        Y = self.readout(res6)
        return Y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S)
        self.hid = Mid_Xnet(T*hid_S*4, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S*4, C)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed = self.enc(x)
        _, C_, H_, W_ = embed.shape
        
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)
        

        Y = self.dec(hid)
        Y = Y.reshape(B, T, C, H, W)
        return Y
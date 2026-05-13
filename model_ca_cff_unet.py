# model_ca_cff_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAtt(nn.Module):
    """
    坐标注意力机制 Coordinate Attention
    适合遥感分割中具有方向性、条带状、扇形边界的目标。
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class DoubleConvCA(nn.Module):
    """
    U-Net基础卷积块，可选择是否加入坐标注意力。
    """

    def __init__(self, in_channels, out_channels, use_ca=True):
        super(DoubleConvCA, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.ca = CoordAtt(out_channels) if use_ca else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)
        return x


class CrossScaleFusion(nn.Module):
    """
    跨尺度特征融合模块：
    high_feat：深层语义特征
    low_feat ：浅层空间细节特征
    """

    def __init__(self, high_channels, low_channels, out_channels):
        super(CrossScaleFusion, self).__init__()

        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CoordAtt(out_channels)
        )

    def forward(self, high_feat, low_feat):
        if high_feat.shape[2:] != low_feat.shape[2:]:
            high_feat = F.interpolate(
                high_feat,
                size=low_feat.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        high_feat = self.high_proj(high_feat)
        low_feat = self.low_proj(low_feat)

        fused = torch.cat([high_feat, low_feat], dim=1)
        fused = self.fuse(fused)

        return fused


class CA_CFF_UNet(nn.Module):
    """
    CA-CFF-UNet：
    融合坐标注意力与跨尺度特征融合的冲积扇提取模型。

    in_channels:
        RGB + DEM + Slope + Hillshade = 6 时设为 6
    out_channels:
        二分类冲积扇提取设为 1
    """

    def __init__(self, in_channels=6, out_channels=1, features=64):
        super(CA_CFF_UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = DoubleConvCA(in_channels, features, use_ca=False)
        self.encoder2 = DoubleConvCA(features, features * 2, use_ca=True)
        self.encoder3 = DoubleConvCA(features * 2, features * 4, use_ca=True)
        self.encoder4 = DoubleConvCA(features * 4, features * 8, use_ca=True)

        self.bottleneck = DoubleConvCA(features * 8, features * 16, use_ca=True)

        self.fuse4 = CrossScaleFusion(features * 16, features * 8, features * 8)
        self.fuse3 = CrossScaleFusion(features * 8, features * 4, features * 4)
        self.fuse2 = CrossScaleFusion(features * 4, features * 2, features * 2)
        self.fuse1 = CrossScaleFusion(features * 2, features, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.fuse4(bottleneck, enc4)
        dec3 = self.fuse3(dec4, enc3)
        dec2 = self.fuse2(dec3, enc2)
        dec1 = self.fuse1(dec2, enc1)

        logits = self.final_conv(dec1)
        return logits
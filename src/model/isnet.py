import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

bce_loss = nn.BCELoss(size_average=True)


def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        # print("i: ", i, preds[i].shape)
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(
                target, size=preds[i].size()[2:], mode="bilinear", align_corners=True
            )
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss
    return loss0, loss


fea_loss = nn.MSELoss(size_average=True)
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)


def muti_loss_fusion_kl(preds, target, dfs, fs, mode="MSE"):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        loss = loss + bce_loss(preds[i], target)

    for i in range(0, len(dfs)):
        if mode == "MSE":
            loss = loss + fea_loss(
                dfs[i], fs[i]
            )  ### add the mse loss of features as additional constraints
            # print("fea_loss: ", fea_loss(dfs[i],fs[i]).item())
        elif mode == "KL":
            loss = loss + kl_loss(F.log_softmax(dfs[i], dim=1), F.softmax(fs[i], dim=1))
            # print("kl_loss: ", kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1)).item())
        elif mode == "MAE":
            loss = loss + l1_loss(dfs[i], fs[i])
            # print("ls_loss: ", l1_loss(dfs[i],fs[i]))
        elif mode == "SmoothL1":
            loss = loss + smooth_l1_loss(dfs[i], fs[i])
            # print("SmoothL1: ", smooth_l1_loss(dfs[i],fs[i]).item())

    return loss


# RELu + BatchNorm + DPConv
class RBDP(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(RBDP, self).__init__()
        # kernel 1X1
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)
        # global average pooling
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        # MLP
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_ch, 3)
        self.softmax = nn.Softmax(dim=1)

        # Paramidal convolution
        self.conv3x3 = nn.Conv2d(
            out_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, stride=stride
        )
        self.conv5x5 = nn.Conv2d(
            out_ch, out_ch, 5, padding=2 * dirate, dilation=1 * dirate, stride=stride
        )
        self.conv7x7 = nn.Conv2d(
            out_ch, out_ch, 7, padding=3 * dirate, dilation=1 * dirate, stride=stride
        )

        # kernel 1X1 final
        self.conv1x1_final = nn.Conv2d(3 * out_ch, out_ch, 1)

        # ReLU + BatchNorm
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        hx = x
        # GAP + MLP
        hx = self.GAP(hx)
        hx = self.flatten(hx)
        hx = self.relu(hx)
        hx = self.linear(hx)
        hx = self.softmax(hx)
        # Paramidal convolution
        out1 = self.conv3x3(x)
        out2 = self.conv5x5(x)
        out3 = self.conv7x7(x)

        # channel wise
        out1 = out1 * hx[:, 0].view(-1, 1, 1, 1)
        out2 = out2 * hx[:, 1].view(-1, 1, 1, 1)
        out3 = out3 * hx[:, 2].view(-1, 1, 1, 1)

        # concatenation
        out_fuse = torch.cat((out1, out2, out3), 1)
        # conv1x1_final
        y = self.conv1x1_final(out_fuse)

        return self.relu_s1(self.bn_s1(y + x))


# RELu + BatchNorm + Conv
class RBC(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=None, padding=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1 * dirate,
            dilation=1 * dirate,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode="bilinear")

    return src


### RSU-7 ###
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7, self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = RBDP(in_ch, out_ch, dirate=1)  ## 1 -> 1/2

        self.rebnconv1 = RBDP(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RBDP(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RBDP(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = RBDP(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = RBDP(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = RBDP(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = RBDP(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = RBDP(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = RBDP(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = RBDP(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = RBDP(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RBDP(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RBDP(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        b, c, h, w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = RBC(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RBC(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RBC(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RBC(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = RBC(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = RBC(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = RBC(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RBC(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = RBC(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RBC(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RBC(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RBC(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = RBC(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = RBC(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RBC(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = RBC(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RBC(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RBC(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RBC(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = RBC(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RBC(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RBC(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = RBC(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RBC(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = RBC(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = RBC(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = RBC(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = RBC(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = RBC(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = RBC(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class myrebnconv(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    ):
        super(myrebnconv, self).__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.rl(self.bn(self.conv(x)))


class ISNetGTEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(ISNetGTEncoder, self).__init__()

        self.conv_in = myrebnconv(
            in_ch, 16, 3, stride=2, padding=1
        )  # nn.Conv2d(in_ch,64,3,stride=2,padding=1)

        self.stage1 = RSU7(16, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 32, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 64, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 64, 512)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def compute_loss(self, preds, targets):
        return muti_loss_fusion(preds, targets)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        # side output
        d1 = self.side1(hx1)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return [
            F.sigmoid(d1),
            F.sigmoid(d2),
            F.sigmoid(d3),
            F.sigmoid(d4),
            F.sigmoid(d5),
            F.sigmoid(d6),
        ], [hx1, hx2, hx3, hx4, hx5, hx6]


class ISNetDIS(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ISNetDIS, self).__init__()

        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return (
            d0,
            [
                F.sigmoid(d1),
                F.sigmoid(d2),
                F.sigmoid(d3),
                F.sigmoid(d4),
                F.sigmoid(d5),
                F.sigmoid(d6),
            ],
            [hx1d, hx2d, hx3d, hx4d, hx5d, hx6],
        )

    def compute_loss_kl(self, preds, targets, dfs, fs, mode="MSE"):
        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)


from torchsummary import summary

# model = ISNetDIS().cuda()
# print(summary(model, (3, 1024, 1024), batch_size=1))

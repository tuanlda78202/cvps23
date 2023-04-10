import torch.nn as nn 
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

# ISNet
bce_loss = nn.BCELoss(size_average=True)
fea_loss = nn.MSELoss(size_average=True)
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)

def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0,len(preds)):
        # print("i: ", i, preds[i].shape)
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target)
        if(i==0):
            loss0 = loss
    return loss0, loss

def muti_loss_fusion_kl(preds, target, dfs, fs, mode='MSE'):
    loss0 = 0.0
    loss = 0.0

    for i in range(0,len(preds)):
        # print("i: ", i, preds[i].shape)
        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target)
        if(i==0):
            loss0 = loss

    for i in range(0,len(dfs)):
        if(mode=='MSE'):
            loss = loss + fea_loss(dfs[i],fs[i]) ### add the mse loss of features as additional constraints
            # print("fea_loss: ", fea_loss(dfs[i],fs[i]).item())
        elif(mode=='KL'):
            loss = loss + kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1))
            # print("kl_loss: ", kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1)).item())
        elif(mode=='MAE'):
            loss = loss + l1_loss(dfs[i],fs[i])
            # print("ls_loss: ", l1_loss(dfs[i],fs[i]))
        elif(mode=='SmoothL1'):
            loss = loss + smooth_l1_loss(dfs[i],fs[i])
            # print("SmoothL1: ", smooth_l1_loss(dfs[i],fs[i]).item())

    return loss0, loss
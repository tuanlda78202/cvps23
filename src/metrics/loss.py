import torch.nn as nn
import torch.nn.functional as F

bce_loss = nn.BCELoss(reduction="mean")
feat_loss = nn.MSELoss(reduction=True)


# U2Net
def multi_bce_fusion(output, mask):
    # Output = [De1, De2, De3, De4, De5, De6, D0]
    loss = 0.0
    for idx in range(0, len(output)):
        loss += bce_loss(output[idx], mask)

    return loss


# DIS (l_fs + l_sg)
def multi_mse_fusion(output, mask, dfs, fs):
    seg_loss = multi_bce_fusion(output, mask)

    for i in range(0, len(dfs)):
        # MSE loss of features as additional constraints
        loss = seg_loss + feat_loss(dfs[i], fs[i])

    return loss

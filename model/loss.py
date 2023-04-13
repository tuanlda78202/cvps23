import torch.nn as nn 
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

# U2Net 
bce_loss = nn.BCELoss(size_average=True)

def multi_bce_fusion(output, mask):
	# Output = [d0, d1, d2, d3, d4, d5, d6]
	loss0 = bce_loss(output[0],mask)
	loss1 = bce_loss(output[1],mask)
	loss2 = bce_loss(output[2],mask)
	loss3 = bce_loss(output[3],mask)
	loss4 = bce_loss(output[4],mask)
	loss5 = bce_loss(output[5],mask)
	loss6 = bce_loss(output[6],mask)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss0, loss
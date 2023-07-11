import gc

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from functools import partial
import wandb
import tqdm
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Pred_endecoder
from model.ebm_models import EBM_Prior
from data import get_loader
from utils import adjust_lr, AvgMeter
from utils import linear_annealing
from utils import visualize_pred_post, visualize_pred_prior, visualize_gt
from loss import *
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
from tools import *
from metrics import maxfm, mae, sm, wfm, em

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["WANDB_MODE"] = "online"

print("Dit cu m dang chjay awfawaow")
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='data/img/', help='Image location')
parser.add_argument('--mask_path', type=str, default='data/img/', help='Mask location')
parser.add_argument('--exp_name', type=str, default='EBM-VAE', help='Name of experiment')
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--save_per_epoch', type=int, default=1, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--lr_ebm', type=float, default=1e-4, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
parser.add_argument('--num_workers', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--focal_lamda', type=int, default=1, help='lamda of focal loss')
parser.add_argument('--bnn_steps', type=int, default=6, help='BNN sampling iterations')
parser.add_argument('--lvm_steps', type=int, default=6, help='LVM sampling iterations')
parser.add_argument('--pred_steps', type=int, default=6, help='Predictive sampling iterations')
parser.add_argument('--smooth_loss_weight', type=float, default=0.4, help='weight of the smooth loss')
parser.add_argument('--ebm_out_dim', type=int, default=1, help='ebm initial sigma')
parser.add_argument('--ebm_middle_dim', type=int, default=60, help='ebm initial sigma')
parser.add_argument('--latent_dim', type=int, default=32, help='ebm initial sigma')
parser.add_argument('--e_init_sig', type=float, default=1.0, help='ebm initial sigma')
parser.add_argument('--e_l_steps', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--e_l_step_size', type=float, default=0.4, help='ebm initial sigma')
parser.add_argument('--e_prior_sig', type=float, default=1.0, help='ebm initial sigma')
parser.add_argument('--g_l_steps', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='ebm initial sigma')
parser.add_argument('--g_l_step_size', type=float, default=0.1, help='ebm initial sigma')
parser.add_argument('--e_energy_form', type=str, default='identity', help='ebm initial sigma')

parser.add_argument('--lat_weight', type=float, default=10.0, help='weight for latent loss')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')

opt = parser.parse_args()
compute_energy = partial(compute_energy_form, e_energy_form=opt.e_energy_form)


print('Generator Learning Rate: {}'.format(opt.lr_gen))

# build models
# build generator
generator = Pred_endecoder(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

# build energy-based prior
ebm_model = EBM_Prior(opt.ebm_out_dim, opt.ebm_middle_dim, opt.latent_dim)
ebm_model.cuda()
ebm_model_params = ebm_model.parameters()
ebm_model_optimizer = torch.optim.Adam(ebm_model_params, opt.lr_ebm)

print("Model based on {} have {:.4f}Mb paramerters in total".format('Generator', sum(x.numel()/1e6 for x in generator.parameters())))
print("EBM based on {} have {:.4f}Mb paramerters in total".format('EBM', sum(x.numel()/1e6 for x in ebm_model.parameters())))

# Dataloader
image_root = opt.image_path
gt_root = opt.mask_path

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
total_step = len(train_loader)

# Loss
CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)

structure_loss = structure_loss_form
structure_loss_focal_loss = partial(structure_loss_focal_loss_form, focal_lamda=opt.focal_lamda)


# logger tools
wandb.login(key="e77e2528122948ecb9e5f3edc70db7927b0770aa")


def sample_p_0(n=opt.batchsize, sig=opt.e_init_sig):
    return sig * torch.randn(*[n, opt.latent_dim, 1, 1]).to(device)

wandb.init(name=opt.exp_name, dir=".")
for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    ebm_model.train()
    loss_record, loss_record_ebm = AvgMeter(), AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    print('EBM Learning Rate: {}'.format(ebm_model_optimizer.param_groups[0]['lr']))
    for i, pack in tqdm.tqdm(enumerate(train_loader, start=1)):
        generator_optimizer.zero_grad()
        ebm_model_optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()
        # multi-scale training samples
        trainsize = int(round(opt.trainsize / 32) * 32)

        z_prior0 = sample_p_0(n=images.shape[0])
        z_post0 = sample_p_0(n=images.shape[0])

        z_e_0, z_g_0 = generator(images, z_prior0, z_post0, gts, prior_z_flag=True, istraining = True)
        z_e_0 = torch.unsqueeze(z_e_0,2)
        z_e_0 = torch.unsqueeze(z_e_0, 3)

        z_g_0 = torch.unsqueeze(z_g_0, 2)
        z_g_0 = torch.unsqueeze(z_g_0, 3)

        ## sample langevin prior of z
        z_e_0 = Variable(z_e_0)
        z = z_e_0.clone().detach()
        z.requires_grad = True
        for kk in range(opt.e_l_steps):
            en = ebm_model(z)
            z_grad = torch.autograd.grad(en.sum(), z)[0]
            z.data = z.data - 0.5 * opt.e_l_step_size * opt.e_l_step_size * (
                    z_grad + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
            z.data += opt.e_l_step_size * torch.randn_like(z).data
        z_e_noise = z.detach()  ## z_

        ## sample langevin post of z
        z_g_0 = Variable(z_g_0)
        z = z_g_0.clone().detach()
        z.requires_grad = True
        for kk in range(opt.g_l_steps):
            _,_,_,gen_res,_ = generator(images, z_prior0, z, gts, prior_z_flag=False, istraining = True)
            g_log_lkhd = 1.0 / (2.0 * opt.g_llhd_sigma * opt.g_llhd_sigma) * mse_loss(
                torch.sigmoid(gen_res), gts)
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            en = ebm_model(z)
            z_grad_e = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * opt.g_l_step_size * opt.g_l_step_size * (
                    z_grad_g + z_grad_e + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
            z.data += opt.g_l_step_size * torch.randn_like(z).data

        z_g_noise = z.detach()  ## z+

        _,_,pred_prior, pred_post, latent_loss = generator(images, z_e_noise, z_g_noise, gts, prior_z_flag=False, istraining = True)
        reg_loss = l2_regularisation(generator.enc_x) + \
                   l2_regularisation(generator.enc_xy) + l2_regularisation(generator.prior_dec) + l2_regularisation(
            generator.post_dec)
        reg_loss = opt.reg_weight * reg_loss
        anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
        loss_latent = opt.lat_weight * anneal_reg * latent_loss
        gen_loss_cvae = opt.vae_loss_weight * (structure_loss(pred_post, gts) + loss_latent)
        gen_loss_gsnn = (1 - opt.vae_loss_weight) * structure_loss(pred_prior, gts)
        loss_all = gen_loss_cvae + gen_loss_gsnn + reg_loss

        loss_all.backward()
        generator_optimizer.step()

        ## learn the ebm
        en_neg = compute_energy(ebm_model(
            z_e_noise.detach())).mean()
        en_pos = compute_energy(ebm_model(z_g_noise.detach())).mean()
        loss_e = en_pos - en_neg
        loss_e.backward()
        ebm_model_optimizer.step()

        visualize_pred_prior(torch.sigmoid(pred_prior), wandb)
        visualize_pred_post(torch.sigmoid(pred_post), wandb)
        visualize_gt(gts, wandb)

        loss_record.update(loss_all.data, opt.batchsize)
        loss_record_ebm.update(loss_e.data, opt.batchsize)

        wandb.log({"Epoch": epoch, "step": i})
        if i % 10 == 0 or i == total_step:
            pred_mask = torch.sigmoid(pred_prior).data.cpu().numpy().squeeze()
            pred_mask = 255 * (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
            pred_mask = pred_mask.astype(np.uint8)

            ground_truth =  gts.data.cpu().numpy().squeeze()
            ground_truth = 255 * (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)
            ground_truth = ground_truth.astype(np.uint8)

            MAXFM = 0
            MAE = 0
            WFM = 0
            SM = 0
            EM = 0
            for i in range(pred_mask.shape[0]):
                # MAXFM += maxfm(pred_mask[i], ground_truth[i])
                MAE += mae(pred_mask[i], ground_truth[i])
                WFM += wfm(pred_mask[i], ground_truth[i])
                SM += sm(pred_mask[i], ground_truth[i])
                EM += em(pred_mask[i], ground_truth[i])

            wandb.log({
                # "max_fm": MAXFM / pred_mask.shape[0],
                "Mean absolute error": MAE / pred_mask.shape[0],
                "Weight F-measure": WFM / pred_mask.shape[0],
                "S-measure": SM / pred_mask.shape[0],
                "Mean E-measure": EM / pred_mask.shape[0]
                }
            )

            del pred_mask, ground_truth
            gc.collect()

        if i % 10 == 0 or i == total_step:
            wandb.log({"generator loss": loss_record.show(), "ebm loss": loss_record_ebm.show()})

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(ebm_model_optimizer, opt.lr_ebm, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.save_per_epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        torch.save(ebm_model.state_dict(), save_path + 'Model' + '_%d' % epoch + '_ebm.pth')

wandb.finish()
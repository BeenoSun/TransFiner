# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _nms, _topk
import torch.nn.functional as F
from utils.image import draw_umich_gaussian

class FusionLoss(nn.Module):

  def __init__(self, opt=None):
    super(FusionLoss, self).__init__()
    self.opt = opt
    self.foca = FastFocalLoss()
  def forward(self, hat, gt, batch):
    '''
    Arguments:
      out, target: B x C x H x W
      mask: B x M
    '''
    mask = batch['mask']
    hat_over_gt_ratio = hat - gt
    #if self.opt.withamp:
    hat_over_gt_ratio[hat_over_gt_ratio >= 1] = 1 - 1e-5
    hat_over_gt_ratio[hat_over_gt_ratio <= -1] = -1 + 1e-5
    #hat_over_gt = out / target
    # version 1: with the weighted absolute hm loss
    # #focus = torch.pow(1 - hat_over_gt, self.opt.loss_second_alpha)
    # focus = torch.abs(1 - hat_over_gt)
    # hm_weight = torch.pow(batch['hm'], self.opt.loss_second_beta)
    # loss = -(hm_weight+self.opt.hm_lift) * focus * torch.log(1 - torch.pow(1 - hat_over_gt, 2) / 9.)
    # num_pos = mask.sum()
    # if num_pos == 0:
    #   return loss.sum()
    # return loss.sum() # loss.sum() / num_pos


    # version 2: with the equal-weight quadratic hm loss also divided by number of objects
    #hm_weight = torch.pow(batch['hm'], self.opt.loss_second_beta)
    #focus = torch.pow(1 - hat_over_gt, 2)
    #loss = -(hm_weight+self.opt.hm_lift) * focus * torch.log(1 - focus / 9.)
    # num_pos = mask.sum()
    # if num_pos == 0:
    #     return loss.sum()

    #pr_gt = (torch.abs(1 - target) + torch.abs(1 - 1. / target)) / 1.5
    #focus_pr = torch.pow(1 - (pr_hat + 1.) / (pr_gt + 1.), 2)
    #loss_pr = -(hm_weight + self.opt.hm_lift) * focus_pr * torch.log(1 - focus_pr / 1.)

    # self, out, target, ind, mask, cat
    hm_fus_loss = focalhogloss(hat_over_gt_ratio, batch['hm'], batch['ind'], mask, batch['cat'])

    return hm_fus_loss

class HMCorrLoss(nn.Module):

  def __init__(self, opt=None):
    super(HMCorrLoss, self).__init__()
    self.opt = opt
    self.foca = FastFocalLoss()
    self.reg = RegWeightedL1Loss()
  def forward(self, out, out_resi, target_resi, batch, wh_):
    '''
    self.loss_second(hm_corr_hat, hm_corr_gt, hm_pr_hat, hm_pr_gt, batch, outputs_copy)
    Arguments:
      out, target: B x C x H x W
      mask: B x M
    '''
    '''
    hat_over_gt = out - target
    hat_over_gt_resi = out_resi - target_resi
    #hat_over_gt_resi[hat_over_gt_resi >= 1] = 1 - 1e-3
    #hat_over_gt_resi[hat_over_gt_resi <= -1] = -1 + 1e-3

    # version 1: with the weighted absolute hm loss
    # #focus = torch.pow(1 - hat_over_gt, self.opt.loss_second_alpha)
    # focus = torch.abs(1 - hat_over_gt)
    # hm_weight = torch.pow(batch['hm'], self.opt.loss_second_beta)
    # loss = -(hm_weight+self.opt.hm_lift) * focus * torch.log(1 - torch.pow(1 - hat_over_gt, 2) / 9.)
    # num_pos = mask.sum()
    # if num_pos == 0:
    #   return loss.sum()
    # return loss.sum() # loss.sum() / num_pos


    # version 2: with the equal-weight quadratic hm loss also divided by number of objects
    #hm_weight = torch.pow(batch['hm'], self.opt.loss_second_beta)
    focus = torch.pow(hat_over_gt, 2)
    obj_mask = batch['hm'] > 0
    n2p_mask = (batch['hm_regu_radius'] > (self.opt.train_thres+0.24)) & (pred[0]['hm'] < (self.opt.train_thres+0.2))
    #p2n_mask = (batch['hm'] < self.opt.train_thres) & (pred[0]['hm'] > (self.opt.train_thres-0.1))
    magic_mask = n2p_mask# | p2n_mask
    #batch['hm'][(batch['hm'] > 0) & (batch['hm'] < 1)] = batch['hm'][(batch['hm'] > 0) & (batch['hm'] < 1)] / 0.3
    loss = focusloss(focus, target, 1, self.opt.stretch_factor, magic_mask)#-focus * torch.log(1 - focus / 9.) # (hm_weight+self.opt.hm_lift) *
    loss = loss / mask.sum() if mask.sum() != 0 else (loss)
    genr_loss = focusloss_lowfp_v2(focus, target, 1, self.opt.stretch_factor, self.opt.alert_fn)
    genr_loss = genr_loss / mask.sum() if mask.sum() != 0 else (genr_loss)
    #pr_gt = (torch.abs(1 - target) + torch.abs(1 - 1. / target)) / 1.5
    #focus_pr = torch.pow(1 - (pr_hat + 1.) / (pr_gt + 1.), 2)
    #loss_pr = -(hm_weight + self.opt.hm_lift) * focus_pr * torch.log(1 - focus_pr / 1.)

    # self, out, target, ind, mask, cat
    focus_resi = torch.pow(hat_over_gt_resi, 2)
    #hm_weight[batch['hm'] == 0] = 1.
    #focus_resi = torch.pow(1 - hat_over_gt_resi, 2)
    #hm_pr_loss = focusloss_lowfp_v2(focus_resi, target_resi, 0, self.opt.stretch_factor, self.opt.alert_fn)#-focus_resi * torch.log(1 - focus_resi)
    #hm_pr_loss = focalhogloss(hat_over_gt_resi, batch['hm'], batch['ind'], mask, batch['cat'])

    #num_pos = mask.sum()
    #if num_pos == 0:
    return (loss + genr_loss),\
           loss, genr_loss
    #else:
      #return (hm_pr_loss)/num_pos, loss/num_pos, hm_pr_loss/num_pos
    '''
    '''bac['negloss_fn_gt'] = negloss_fn_gt
    bac['fn_mask'] = fn_mas
    bac['negloss_fp_gt'] = negloss_fp_gt
    bac['fp_mas'] = fp_mas'''
    mask = batch['mask']
    # fn
    gt = torch.pow(1 - batch['negloss_fn_gt'], 4)
    #out_pos = ((1 / (1 + target[target>0].min() - target)) * (1. - out))[batch['fn_mask'] != 0]
    #out_pos[out_pos >= 1] = 1 - 1e-4
    #out_pos_focu = (1. - out)[batch['fn_mask'] != 0]
    out_pos = (out)[batch['fn_mask'] != 0]
    out_neg = out[batch['fn_mask'] == 0]
    neg_loss = torch.log(1 - out_neg) * torch.pow(out_neg, 2) * gt[batch['fn_mask'] == 0]
    neg_loss = neg_loss.sum()
    num_pos = mask.sum()
    pos_loss = torch.log(out_pos) * torch.pow(1. - out_pos, 2)
    pos_loss = pos_loss.sum()

    # fp
    gt_fp = torch.pow(1 - batch['negloss_fp_gt'], 4)
    out_pos_fp = (out_resi)[batch['fp_mask'] != 0]
    out_neg_fp = out_resi[batch['fp_mask'] == 0]
    neg_loss_fp = torch.log(1 - out_neg_fp) * torch.pow(out_neg_fp, 2) * gt_fp[batch['fp_mask'] == 0]
    neg_loss_fp = neg_loss_fp.sum()
    pos_loss_fp = torch.log(out_pos_fp) * torch.pow(1. - out_pos_fp, 2)
    pos_loss_fp = pos_loss_fp.sum()
    if num_pos == 0:
        return - neg_loss, - neg_loss_fp
    return - (pos_loss + neg_loss), - (pos_loss_fp + neg_loss_fp)


def focusloss_lowfp_v2(focus, gt, divide, stretch_factor, alert_fn):
    focus_pos = focus[gt >= divide]
    focus_neg = focus[gt < divide]
    #focus_pos_streng = focus[(gt-divide) > alert_fn] * stretch_factor
    focus_pos[focus_pos >= 1] = 1 - 1e-4
    focus_neg[focus_neg >= 1] = 1 - 1e-4
    #focus_pos_streng[focus_pos_streng >= 1] = 1 - 1e-4
    return -((focus_pos * torch.log(1 - focus_pos)).sum() + (focus_neg * torch.log(1 - focus_neg)).sum())

def focusloss(focus, gt, divide, stretch_factor, magic_mask):
    focus_pos = focus[magic_mask] * stretch_factor
    focus_neg = focus[(gt < divide) & magic_mask]
    focus_pos[focus_pos >= 1] = 1 - 1e-4
    focus_neg[focus_neg >= 1] = 1 - 1e-4
    return -((focus_pos * torch.log(1 - focus_pos)).sum())

def focalhogloss(focus_, hm, ind, mask, cat):
    '''
    Arguments:
        focus_resi, batch['ind'], mask, batch['cat']
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    gt = torch.pow(1 - hm, 2)
    # opt.w_refinemap *
    focus = focus_.clone()

    focus[focus >= 1] = 1 - 1e-4
    focus[focus <= -1] = -1 + 1e-4
    neg_loss = torch.log(1 - torch.abs(focus)) * torch.pow(focus, 2) * gt
    neg_loss = neg_loss.sum()
    pos_pred_pix = _tranpose_and_gather_feat(focus, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(1 - torch.abs(pos_pred)) * torch.pow(pos_pred, 2) * \
           mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss * 50
    return - (pos_loss + neg_loss) / num_pos * 50

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _neg_loss(pred, gt):
  ''' Reimplemented focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _only_neg_loss(pred, gt, second_stage=False):
  gt = torch.pow(1 - gt, 4)
  if second_stage:
      pred[(pred==1-1e-4) & (gt<1)] = 1-1e-3
      # opt.w_refinemap *
      neg_loss = torch.log(1 - pred + 1e-5) * torch.pow(pred, 2) * gt * 0.3
  else:
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self, opt=None):
    super(FastFocalLoss, self).__init__()
    self.only_neg_loss = _only_neg_loss

  def forward(self, out, target, ind, mask, cat, second_stage=False):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    neg_loss = self.only_neg_loss(out, target, second_stage)
    pos_pred_pix = _tranpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    if second_stage:
        pos_pred[pos_pred==1e-4] = 1e-3
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * 0.3 *\
                   mask.unsqueeze(2)
    else:
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) if second_stage else - (pos_loss + neg_loss) / num_pos

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss


class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class WeightedBCELoss(nn.Module):
  def __init__(self):
    super(WeightedBCELoss, self).__init__()
    self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, output, mask, ind, target):
    # output: B x F x H x W
    # ind: B x M
    # mask: B x M x F
    # target: B x M x F
    pred = _tranpose_and_gather_feat(output, ind) # B x M x F
    loss = mask * self.bceloss(pred, target)
    loss = loss.sum() / (mask.sum() + 1e-4)
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
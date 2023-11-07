import cv2
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from post_transfiner.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from utils.debugger import Debugger
import math
from utils.post_process import generic_post_process
from utils.image import draw_umich_gaussian, gaussian_radius
import torch.nn.functional as F
from torchvision.ops import nms, box_iou
from model.utils import _gather_feat, _tranpose_and_gather_feat
from model.utils import _nms, _topk, _topk_channel
from opts import opts
opt = opts().parse()


gt_total_rate_ = 0
total_valid_rate_ = 0
area_ = 0
num_ = 0

def patchfilter(patch_box, matched):
    """
    filter out the same patch index in each img!
    Args:
        patch_box: bsz x num_patch_query x 4   'center_x, center_y, w, h'
        matched: a match list contains bsz tuples indicating the row_ind and col_ind

    Returns:
        patched_box: patched area to be sent to decoder for re-detection, size: num_patch2decoder x 4
                     in the form of 'center_x, center_y, w, h'
    """
    patched_box = []
    if len(matched) == 0:
        for i in range(patch_box.shape[0]):
            patched_box.append(torch.tensor([[0.5, 0.5, 1., 1.]], device=patch_box.device))
    else:
        for i, mach in enumerate(matched):
            if min(mach[0].shape) == 0:
                patched_box.append(torch.tensor([[0.5, 0.5, 1., 1.]], device=patch_box.device))
            else:
                patched_box.append(patch_box[i, mach[0], :])

    return torch.cat(patched_box, dim=0)


def hmbased_initialization(patch2batch, patch_area, batc, memory, spatial_shapes):
    """patch2batch, patch_area, batc, memory, spatial_shapes

    Args:
        patch2batch: a length of num_patch2decoder list matches the patch to the corresponding img
        patch_area (sigmoided): patched area to be sent to decoder for re-detection (sometimes there are several for a poor prediction)
                     in the form of 'topx, topy, botx, boty'
        batc: batch
        memory:
        spatial_shapes:
        mask_flatten:
        hmprob_init: True when choosing probability initialization; False for topk initialization
    Returns:

    """

    with torch.no_grad():
        # select a base patch size
        resc_h, resc_w = spatial_shapes[0]
        # 0 for object; 1 for objectless    bsz x 1 x 2resc_h x 2resc_w
        mask_re = F.interpolate(batc['pad_mask'][:, None].float(),
                             size=[opt.input_h // 4, opt.input_w // 4]).detach().clone().int() # /4

        # patch_area_rate : (top_x, top_y, bottom_x, bottom_y) # /4
        patch_area = (patch_area.detach().clone() *
                      torch.tensor([2 * resc_w, 2 * resc_h, 2 * resc_w, 2 * resc_h], device=patch_area.device)[None]).round().int()

        # initialize the queries and refer points with hm as prob map
        init_reference_out, pre_init_reference_out = \
            torch.zeros((patch_area.shape[0], opt.real_num_queries, 6), device=memory.device), \
            torch.zeros((patch_area.shape[0], opt.real_num_queries, 6), device=memory.device)

        mask_flatten_rec, src_valid_trans = [], []
        query_recon_masks, pred_gt_mats = [], []
        for patch_i in range(patch_area.shape[0]):
            mask_flatten_re = []
            patch_topx, patch_topy, patch_bottom_x, patch_bottom_y = patch_area[patch_i]
            patch_h, patch_w = patch_bottom_y - patch_topy, patch_bottom_x - patch_topx

            mask_pat = mask_re[patch2batch[patch_i], :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]

            # to get the trans matrixs for valid patch of proposed patch for sigmoid predictions
            # mask the region out of the patch
            patch_mask = torch.ones(list(mask_re.shape)[1:])
            patch_mask[:, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x] = 0
            mask_re_copy = mask_re[patch2batch[patch_i]].detach().clone()
            mask_re_copy[patch_mask.to(torch.bool)] = 1

            # topk & semi-random init for transformer
            if ('iscrowdhuman' in batc) and batc['iscrowdhuman'][patch_i]:
                propos, query_recon_mask, pred_gt_mat = \
                    fake_track(batc, patch_area, patch_i, mask_pat)
            else:
                propos, query_recon_mask, pred_gt_mat = \
                    tracker_extract_disturb(batc, patch_area, patch_i, mask_pat)

            '''if not opt.eva:
                propos, query_recon_mask, pred_gt_mat = \
                    fake_track(batc, patch_area, patch_i, mask_pat)
            else:
            # topk & semi-random init for transformer
            propos, query_recon_mask, pred_gt_mat = \
                tracker_extract_disturb(batc, patch_area, patch_i, mask_pat)'''

            if len(propos) == 0:
                src_valid_trans.append(
                    torch.zeros(opt.num_feature_levels, 4, device=memory.device))  # 4 x 2 x 2 x 3
                init_reference_out[patch_i, :, :] = torch.zeros(opt.real_num_queries, 6, device=memory.device)  # num_query x 2
                pre_init_reference_out[patch_i, :, :] = torch.zeros(opt.real_num_queries, 6, device=memory.device)
                mask_flatten_rec.append(torch.ones(1, memory.shape[1], device=memory.device, dtype=torch.bool))  # 1 x 10845 bool
                query_recon_masks.append(torch.zeros(opt.real_num_queries, device=memory.device).bool())
                pred_gt_mats.append((torch.zeros(opt.real_num_queries) - 1).to(memory.device))
                continue
            res_trans = torch.tensor([opt.ori_input_w / opt.input_w, opt.ori_input_h / opt.input_h], device=mask_pat.device)
            src_valid_tran = get_transes_patchlvl(mask_re_copy, resc_h, resc_w)  # lvls x 4
            src_valid_trans.append(src_valid_tran)

            query_recon_masks.append(query_recon_mask)
            pred_gt_mats.append(pred_gt_mat)
            sampled_xy = propos['cts'][0]
            init_reference_out[patch_i, :, :2] = sampled_xy / torch.stack([patch_w, patch_h], dim=0) / res_trans
            init_reference_out[patch_i, :, 2:] = torch.clip(propos['bboxes_amodal'][0] / torch.stack([patch_w, patch_h, patch_w, patch_h], dim=0) / res_trans.repeat(2),
                                                            min=0.01, max=torch.maximum(2*resc_w/patch_w, 2*resc_h/patch_h))
            # pre
            pre_sampled_xy = propos['pre_cts'][0]
            pre_init_reference_out[patch_i, :, :2] = pre_sampled_xy / torch.stack([patch_w, patch_h], dim=0) / res_trans
            pre_init_reference_out[patch_i, :, 2:] = torch.clip(
                                propos['pre_bboxes_amodal'][0] / torch.stack([patch_w, patch_h, patch_w, patch_h], dim=0) / res_trans.repeat(2),
                                min=0.01, max=torch.maximum(2 * resc_w / patch_w, 2 * resc_h / patch_h))

            # to get the updated mask for decoder
            for lvl_i in range(opt.num_feature_levels):
                mask = F.interpolate(mask_re_copy[None].float(), size=spatial_shapes[lvl_i].cpu().tolist()).to(torch.bool)[0]
                mask = mask.flatten(1)
                mask_flatten_re.append(mask)
            mask_flatten_rec.append(torch.cat(mask_flatten_re, 1))

        return init_reference_out, pre_init_reference_out, \
               torch.cat(mask_flatten_rec, 0), torch.stack(src_valid_trans), \
               torch.stack(query_recon_masks, 0), torch.stack(pred_gt_mats, 0)


def get_transes_patchlvl(mask_pat, patch_h, patch_w, lvls=4):
    """

    Args:
        mask_pat: 1 x patch_h x patch_w   0 for object; 1 for objectless  int
        patch_h:
        patch_w:

    Returns:
        src_valid_trans: lvls x 4
    """

    src_valid_trans = []
    patch_hws = [[(patch_h/2**sca).ceil().int() if (patch_h/2**sca).ceil().int() > 1 else torch.tensor(2,device=mask_pat.device,dtype=torch.int32),
                  (patch_w/2**sca).ceil().int() if (patch_w/2**sca).ceil().int() > 1 else torch.tensor(2,device=mask_pat.device,dtype=torch.int32)]
                  for sca in range(lvls)]

    for patch_hw in patch_hws:
        mask = F.interpolate(mask_pat[None].float(), size=patch_hw).to(torch.bool)[0]
        if torch.nonzero(~mask[0]).shape[0] == 0:
            src_valid_trans.append(src_valid_trans[-1])
            continue
        tl, br = torch.nonzero(~mask[0])[[0, -1]]
        bl = tl * torch.tensor([0, 1], device=tl.device) + br * torch.tensor([1, 0], device=tl.device)
        if (tl[0] == br[0]) or (br[1] == bl[1]):
            src_valid_trans.append(src_valid_trans[-1])
            continue
        patch_area = torch.cat([tl.flip(-1) / torch.stack(patch_hw).flip(-1), ((br.flip(-1)+1) / torch.stack(patch_hw).flip(-1))])
        src_valid_trans.append(patch_area)

    return torch.stack(src_valid_trans).to(mask_pat.device)


def pre2samp(refer_point_sig, src_valid_trans, bac=None, recti_wh=None, lvls=None):
    """

    Args:
        refer_point_sig: psz x n_q x 2 or psz x n_q x 4/6  ranging from 0 ~ 1
        src_valid_trans: psz x lvls x 4   'topx, topy, botx, boty'
        recti_wh: lvls x 2   int_w, int_y
    Returns:
        samp_points: psz x n_q x lvls x 2
    """
    patch_whs = torch.stack([(recti_wh / 2 ** sca).ceil().int() for sca in range(lvls)]) # lvls x 2
    patch_topxy_rectifier = ((src_valid_trans[..., :2] * patch_whs[None]).int() + 1e-5) / (src_valid_trans[..., :2] * patch_whs[None] + 1e-5) # bsz x lvls x 2
    patch_wh_rectifier = ((src_valid_trans[..., 2:]*patch_whs[None]).int() - (src_valid_trans[..., :2]*patch_whs[None]).int() + 1e-5) \
                         / ((src_valid_trans[..., 2:] - src_valid_trans[..., :2]) * patch_whs[None] + 1e-5) # bsz x lvls x 2
    if refer_point_sig.shape[-1] == 2:
        # -> psz x n_q x lvls x 2
        samp_points = refer_point_sig[:, :, None]
        samp_points = (src_valid_trans[..., :2] * patch_topxy_rectifier)[:, None] + \
                      ((src_valid_trans[..., 2:] - src_valid_trans[..., :2]) * patch_wh_rectifier)[:, None] * samp_points # bsz x n_q x lvls x 2
    else:
        assert refer_point_sig.shape[-1] == (4 if not opt.box_amodal_pred else 6)
        cts, amodals = torch.split(refer_point_sig, [2, 4], dim=-1) # bsz x n_q x 2/4
        samp_points = torch.cat([cts[:, :, None], amodal2points(amodals, bac['nnpp'], cts)], dim=-2) # bsz x n_q x 5 x 2
        # -> psz x n_q x lvls x 5 x 2
        samp_points = (src_valid_trans[..., :2] * patch_topxy_rectifier)[:,None,:,None] + \
                      ((src_valid_trans[..., 2:] - src_valid_trans[..., :2]) * patch_wh_rectifier)[:,None,:,None] * samp_points[:,:,None]
        # -> psz x n_q x lvls x 1/4 x 2
        cts, amodals = torch.split(samp_points, [1, 4], dim=-2)
        # -> psz x n_q x lvls x 4
        amodals = points2amodal(amodals, bac['nnpp'], cts.squeeze(-2))
        # -> psz x n_q x lvls x 6
        samp_points = torch.cat([cts.squeeze(-2), amodals], dim=-1)

    return samp_points


def pre2samp_inv(refer_point, src_valid_trans):
    """

    Args:
        refer_point:  psz x n_q x 2
        src_valid_trans: psz x lvls x nor/inv x 2 x 3    nor: patch          -> valid subpatch
                                                         inv: valid subpatch -> patch

    Returns:
        refer_point_sig: psz x n_q x 2    ranging from 0 ~ 1
    """
    if refer_point.shape[-1] == 2:
        refer_point = torch.cat([refer_point[:, :, :, None], torch.ones(refer_point.shape[:-1],
                                device=refer_point.device)[..., None, None]], dim=-2)  # bsz x n_q x 3 x 1
        refer_point_sig = torch.matmul(src_valid_trans[:, 0, 1][:, None].float(), refer_point).squeeze(-1).float()  # bsz x n_q x 2
    else:
        assert refer_point.shape[-1] == (4 if not opt.box_amodal_pred else 6)
        nnpp = torch.tensor([-1, -1, 1, 1], device=refer_point.device)
        ct = refer_point[..., :2]
        bbox_amodal = refer_point[..., 2:]
        tran = src_valid_trans[:, 0:1]
        rect_points = amodal2points(bbox_amodal, nnpp, ct)
        for i in range(rect_points.shape[-2]):
            rect_points[:, :, i] = pre2samp_inv(rect_points[:, :, i], tran)
        ct = pre2samp_inv(ct, tran)
        bbox_amodal = points2amodal(rect_points, nnpp, ct)
        refer_point_sig = torch.cat([ct, bbox_amodal], dim=-1)

    return refer_point_sig


def generate_refine_gt(batc, patch_area, patch2batch):
    """

    Args:
        batc['anns']['bbox']: topx, topy, w, h            bsz x 100 x 4 : padded with zeros for group of imgs
        patch_area (sigmoided): topx, topy, botx, boty    num_patch2decoder x 4
        patch2batch: list length of num_patch2decoder     matching each patch to the corresponding img

    Returns:
        refine_targets -> targ['boxes']: 'center_x, center_y, w, h'   'center_x, center_y' are transed to 0 ~ 1
    """

    batc['refine_targets'] = []
    batc['refine_targets'].append({'patch_area': patch_area.detach().clone()})

    # converting -> topx, topy, botx, boty
    tg_bbox = torch.stack([batc['anns'][..., 0],  batc['anns'][..., 1],
                           batc['anns'][..., 0] + batc['anns'][..., 2],
                           batc['anns'][..., 1] + batc['anns'][..., 3]], dim=-1) # bsz x 100 x 4
    # rect: [[lt], [lb], [rb], [rt]]
    rect = torch.stack([tg_bbox[..., [0, 1]], tg_bbox[..., [0, 3]],
                        tg_bbox[..., [2, 3]], tg_bbox[..., [2, 1]]], dim=2)

    # test only
    #retr_ind_total = tg_bbox.sum(-1) > 0
    #gt_total_num = retr_ind_total.sum()

    rect = affine_trans(rect, batc['cur_meta']['trans_output'][:, None]) # bsz x 100 x 4 x 2

    # tg_bbox: topx, topy, botx, boty   'bsz x 100 x 4'
    tg_bbox[..., :2] = torch.stack([rect[..., 0].min(-1).values, rect[..., 1].min(-1).values], dim=-1)  # topx, topy
    tg_bbox[..., 2:] = torch.stack([rect[..., 0].max(-1).values, rect[..., 1].max(-1).values], dim=-1)  # botx, boty
    tg_bbox_amodal = tg_bbox.detach().clone()  # 'bsz x 100 x 4'
    tg_ct = torch.stack([torch.clip(tg_bbox_amodal[..., [0, 2]], 0, opt.input_w // 8 - 1).sum(-1) / 2,
                         torch.clip(tg_bbox_amodal[..., [1, 3]], 0, opt.input_h // 8 - 1).sum(-1) / 2],
                        dim=-1)  # 'bsz x 100 x 2'
    tg_bbox_amodal = tg_ct.repeat(1, 1, 2) * (-1 * batc['nnpp']) + tg_bbox_amodal * batc['nnpp']

    tg_bbox = tg_bbox / torch.tensor([opt.input_w // 8, opt.input_h // 8, opt.input_w // 8, opt.input_h // 8],
                                      device=tg_bbox.device)
    batc['targets_xyxy'] = tg_bbox.detach().clone()
    tg_bbox = torch.clip(tg_bbox, 0., 1.)
    # -> tg_bbox: center_x, center_y, w, h   'bsz x 100 x 4'
    tg_bbox = box_xyxy_to_cxcywh(tg_bbox)

    # retrieve the 'tg' in the patch area
    retr_ind = ((tg_bbox[patch2batch, :, :2] > patch_area[:, None, :2]) *
                (tg_bbox[patch2batch, :, :2] < patch_area[:, None, 2:])).prod(-1).bool()  # num_patch2decoder x 100
    retr_ind_valid = (tg_bbox[patch2batch, :, 2:].abs().sum(-1) != 0)
    retr_ind = retr_ind * retr_ind_valid
    # program test only
    '''global area_, num_, gt_total_rate_, total_valid_rate_
    retr_ind_total = (((tg_bbox[retr_ind_total][:, :2]>torch.tensor([0,0],device=tg_bbox.device))
                       *(tg_bbox[retr_ind_total][:, :2]<torch.tensor([1,1], device=tg_bbox.device))).prod(-1)>0).sum()
    area_ = (area_*num_ + (batc['refine_targets'][0]['patch_area'][:, 2:]-batc['refine_targets'][0]['patch_area'][:, :2]).prod(-1).mean())/(num_+1)
    gt_total_rate_ = (gt_total_rate_*num_ + retr_ind.sum() / retr_ind_total) / (num_ + 1)
    total_valid_rate_ = (total_valid_rate_*num_ + retr_ind_total / gt_total_num) / (num_ + 1)
    num_ = num_ + 1
    print('iter:', num_, 'patch area:', area_, 'gt_refine/total_valid:', gt_total_rate_, 'total_valid/total:', total_valid_rate_)'''
    # program test only
    batc['targets_xyxy'] = batc['targets_xyxy'][retr_ind]
    # patch_area : (top_x, top_y, bottom_x, bottom_y)  'num_patch2decoder x 4'
    patch_area = (patch_area.detach().clone() *
                  torch.tensor([opt.input_w // 8, opt.input_h // 8, opt.input_w // 8, opt.input_h // 8],
                               device=patch_area.device)[None]).int()
    batc['refine_targets_beforetrans'] = []
    for num_pat in range(retr_ind.shape[0]):
        patch_topx, patch_topy, patch_bottom_x, patch_bottom_y = patch_area[num_pat]
        patch_wh = patch_area[num_pat:(num_pat + 1), 2:] - patch_area[num_pat:(num_pat + 1), :2]
        mask = F.interpolate(batc['pad_mask'][patch2batch[num_pat]:(patch2batch[num_pat] + 1)][None].float(),
                             size=[opt.input_h // 8, opt.input_w // 8]).to(torch.bool)[0].detach().clone()
        mask_pat = mask[:, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x].int()
        target = {}
        if mask_pat[mask_pat == 0].shape[0] == 0:
            target['boxes'] = torch.tensor([], device=mask.device)
            target['motion'] = torch.tensor([], device=mask.device)
            batc['refine_targets'].append(target)
            continue
        elif mask_pat[mask_pat == 1].shape[0] != 0:
            tl, br = torch.nonzero(~mask_pat[0].bool())[[0, -1]]
            patch_area[num_pat:(num_pat + 1), 2:] = patch_area[num_pat:(num_pat + 1), :2] + br.flip(-1) + 1
            patch_area[num_pat:(num_pat + 1), :2] = patch_area[num_pat:(num_pat + 1), :2] + tl.flip(-1)
            patch_wh = patch_area[num_pat:(num_pat + 1), 2:] - patch_area[num_pat:(num_pat + 1), :2]

        patch_tg_ct = tg_ct[patch2batch[num_pat], retr_ind[num_pat]].detach().clone()
        patch_tg_bbox_amodal = tg_bbox_amodal[patch2batch[num_pat], retr_ind[num_pat]].detach().clone()
        patch_tg_motion = batc['pre_bxs'][patch2batch[num_pat], retr_ind[num_pat]].detach().clone()
        if 0 in list(patch_tg_ct.shape):
            target['boxes'] = torch.tensor([], device=patch_tg_ct.device)
            target['motion'] = torch.tensor([], device=mask.device)
            batc['refine_targets'].append(target)
            continue

        patch_tg_ct = (patch_tg_ct - patch_area[num_pat:(num_pat + 1), :2]) / patch_wh
        patch_tg_bbox_amodal = patch_tg_bbox_amodal / (patch_wh).repeat(1, 2)
        patch_tg_motion[..., :2] = patch_tg_motion[..., :2] - patch_area[num_pat:(num_pat + 1), :2]
        patch_tg_motion = patch_tg_motion / (patch_wh).repeat(1, 3)

        batc['refine_targets_beforetrans'].append(
            torch.cat([patch_tg_ct.detach().clone(), patch_tg_bbox_amodal.detach().clone()], dim=-1))  # program test only

        # tran = get_transes_patchlvl(mask_pat, patch_h, patch_w, lvls=1)
        # rect_points = amodal2points(patch_tg_bbox_amodal, batc['nnpp'], patch_tg_ct)
        # rect_points = pre2samp_inv(rect_points, tran[:, None]).squeeze(0)
        # patch_tg_ct = pre2samp_inv(patch_tg_ct[None], tran[:, None]).squeeze(0)
        # patch_tg_bbox_amodal = points2amodal(rect_points, batc['nnpp'], patch_tg_ct)

        target['boxes'] = torch.cat([patch_tg_ct, patch_tg_bbox_amodal], dim=-1)
        target['motion'] = patch_tg_motion
        batc['refine_targets'].append(target)

    batc['refine_targets'][0]['patch_area'] = \
        patch_area / torch.tensor([opt.input_w // 8, opt.input_h // 8, opt.input_w // 8, opt.input_h // 8], device=patch_area.device).detach().clone()

    return patch_area / torch.tensor([opt.input_w // 8, opt.input_h // 8, opt.input_w // 8, opt.input_h // 8], device=patch_area.device)

def eval_pa(batc, patch_area):
    num_pat = 0
    patch_area = (patch_area.detach().clone() *
                  torch.tensor([opt.input_w // 8, opt.input_h // 8, opt.input_w // 8, opt.input_h // 8],
                               device=patch_area.device)[None]).int()
    patch_topx, patch_topy, patch_bottom_x, patch_bottom_y = patch_area[0]
    mask = F.interpolate(batc['pad_mask'][None].float(),
                         size=[opt.input_h // 8, opt.input_w // 8]).to(torch.bool)[0].detach().clone()
    mask_pat = mask[:, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x].int()
    tl, br = torch.nonzero(~mask_pat[0].bool())[[0, -1]]
    patch_area[num_pat:(num_pat + 1), 2:] = patch_area[num_pat:(num_pat + 1), :2] + br.flip(-1) + 1
    patch_area[num_pat:(num_pat + 1), :2] = patch_area[num_pat:(num_pat + 1), :2] + tl.flip(-1)

    return patch_area / torch.tensor([opt.input_w // 8, opt.input_h // 8, opt.input_w // 8, opt.input_h // 8], device=patch_area.device)

def amodal2points(bbox_amodal, nnpp, ct): # amodaltopx -> topx
    # bbox_amodal: ... x 4
    bbox = bbox_amodal * nnpp + ct.repeat([1 if nu<(len(ct.shape)-1) else 2 for nu in range(len(ct.shape))])
    bbox = torch.stack([bbox[..., [0, 1]], bbox[..., [0, 3]],
                        bbox[..., [2, 3]], bbox[..., [2, 1]]], dim=-2)
    return bbox # ... x 4 x 2

def points2amodal(bbox, nnpp, ct): # topx -> amodaltopx
    # bbox: ... x 4 x 2
    bbox_amodal = torch.cat([torch.stack([bbox[..., 0].min(-1).values, bbox[..., 1].min(-1).values], dim=-1),
                             torch.stack([bbox[..., 0].max(-1).values, bbox[..., 1].max(-1).values], dim=-1)], dim=-1) # ... x 4
    bbox_amodal = bbox_amodal * nnpp - nnpp * ct.repeat([1 if nu<(len(ct.shape)-1) else 2 for nu in range(len(ct.shape))])
    return bbox_amodal

def generate_patch_gt(iou):
    """

    Args:
        iou: n_pred x n_gt

    Returns:

    """
    pred_gt_mat = torch.zeros(opt.real_num_queries, device=iou.device) - 1
    row_ind, col_ind = linear_sum_assignment((1-iou).cpu().numpy())
    for i in range(len(row_ind)):
        if iou[row_ind[i], col_ind[i]] >= opt.reconstruct_thre:
            pred_gt_mat[row_ind[i]] = col_ind[i]

    return pred_gt_mat


def mergpats(patchs):
    patchs[0, :2] = patchs[:, :2].min(0).values
    patchs[0, 2:] = patchs[:, 2:].max(0).values
    return box_xyxy_to_cxcywh(patchs[0:1])


def track_merge(patch_area, ret, output, gt_box=None, tes=False):
    """
    performs the post refine object merging with the original one
    Args:
        patch_area: topx, topy, botx, boty

        ret ->    ret['cts']: 1 x K x 2
            -> ret['bboxes']: 1 x K x 4
            -> ret['scores']: 1 x K

        output -> ['result_refine'] ->    [0]['refine_cts']: num_refine_proposal x 2
                                    ->  [0]['refine_boxes']: num_refine_proposal x 4
                                    -> [0]['refine_scores']: num_refine_proposal

        gt_box : 1 x num_gt x 4
    Returns:

    """
    # select the index of the original need to be further concerned
    abandon_ind = (((ret['cts'][0] > patch_area[:, :2]) *
                   (ret['cts'][0] < patch_area[:, 2:])).prod(-1) *
                   ret['scores'][0] > opt.out_thresh).bool() #

    if not opt.test_transpatch:
        if (output['result_refine'][0]['refine_scores'] > opt.refine_thresh).sum() == 0:
            if tes:
                return ret, abandon_ind.sum(), 0, gt_box.shape[0]
            else:
                ret_ori = {}
                for nam in ret:
                    ret_ori[nam] = ret[nam].clone()
                ret['scores'] = torch.zeros_like(ret['scores'])
                ret['tracking'] = torch.zeros(1, 100, 4).to(ret['scores'].device)
                return ret, ret_ori

    ori_boxes = ret['bboxes'][:, abandon_ind].detach().clone()
    ori_scores = ret['scores'][:, abandon_ind].detach().clone()

    # program test only
    gt_box = gt_box.to(patch_area.device)[0]
    num_gt_total = gt_box.shape[0]
    gt_box_amodal = gt_box.detach().clone()
    gt_box[:, [0, 2]] = torch.clip(gt_box[:, [0, 2]], min=0, max=opt.input_w // 4 - 1)
    gt_box[:, [1, 3]] = torch.clip(gt_box[:, [1, 3]], min=0, max=opt.input_h // 4 - 1)
    gt_box[:, :2] = (gt_box[:, :2] + gt_box[:, 2:]) / 2


    # to test the hw relative to the patch area
    """
    patch_w, patch_h = patch_area[0, 2:] - patch_area[0, :2]
    nnpp = torch.tensor([-1, -1, 1, 1], device=gt_box.device)
    cliped_w = torch.clip((gt_box_amodal*nnpp - nnpp*gt_box[:, :2].repeat(1, 2))[:, [0, 2]], min=0, max=patch_w - 1)
    cliped_h = torch.clip((gt_box_amodal*nnpp - nnpp*gt_box[:, :2].repeat(1, 2))[:, [1, 3]], min=0, max=patch_h - 1)
    amod = torch.stack([cliped_w[:, 0], cliped_h[:, 0], cliped_w[:, 1], cliped_h[:, 1]], dim=-1)
    gt_box_amodal = gt_box[:, :2].repeat(1, 2) + amod * nnpp
    """
    abandon_ind_gt = ((gt_box[:, :2] > patch_area[:, :2]) *
                      (gt_box[:, :2] < patch_area[:, 2:])).prod(-1).bool()  # 100
    v_a = (torch.minimum(gt_box_amodal[abandon_ind_gt][:, 2:], patch_area[:, 2:]) - torch.maximum(gt_box_amodal[abandon_ind_gt][:, :2], patch_area[:, :2])).prod(-1) \
                / (gt_box_amodal[abandon_ind_gt][:, 2:] - gt_box_amodal[abandon_ind_gt][:, :2]).prod(-1)
    num_pr = abandon_ind.sum()
    num_gt = abandon_ind_gt.sum()
    #'''
    ret_ori = {}
    for nam in ret:
        ret_ori[nam] = ret[nam].clone()
    for nam in ret:
        if ret[nam].shape[1] != abandon_ind.shape[0]:
            continue
        ret[nam][:, abandon_ind] = torch.zeros_like(ret[nam][:, abandon_ind])
    #'''
    '''
    #program test only (directly appending gt to the pred)
    refinement = {}
    refinement['bboxes'] = gt_box_amodal[abandon_ind_gt][None]
    refinement['scores'] = torch.ones((1, refinement['bboxes'].shape[1]), dtype=torch.float32,
                                      device=ret['clses'].device)  # n_q
    refinement['bboxes_amodal'] = refinement['bboxes'].detach().clone()
    refinement['cts'] = (gt_box[:, :2][abandon_ind_gt])[None]
    refinement['clses'] = torch.zeros((1, refinement['bboxes'].shape[1]), dtype=torch.float32,
                                      device=ret['clses'].device)
    refinement['tracking'] = torch.zeros((1, refinement['bboxes'].shape[1], 2), dtype=torch.float32,
                                         device=ret['tracking'].device)
    '''
    #'''
    refinement = {}
    refinement['scores'] = output['result_refine'][0]['refine_scores'][output['result_refine'][0]['refine_scores'] > opt.refine_thresh] # n_proposal
    refinement['bboxes'] = (output['result_refine'][0]['refine_boxes'][output['result_refine'][0]['refine_scores'] > opt.refine_thresh] *
                            torch.tensor([opt.input_w // 4, opt.input_h // 4, opt.input_w // 4, opt.input_h // 4],
                            device=patch_area.device))[None] # 1 x n_proposal x 4
    refinement['tracking'] = (output['result_refine'][0]['refine_tracking'][output['result_refine'][0]['refine_scores'] > opt.refine_thresh] *
                              torch.tensor([opt.input_w // 4, opt.input_h // 4, opt.input_w // 4, opt.input_h // 4],
                              device=patch_area.device))[None].squeeze(2)  # 1 x n_proposal x 4

    refine_gt_ious = box_iou(refinement['bboxes'][0], gt_box_amodal[abandon_ind_gt])
    ori_gt_ious = box_iou(ori_boxes[0], gt_box_amodal[abandon_ind_gt])

    refinement['cts'] = (output['result_refine'][0]['refine_cts'][output['result_refine'][0]['refine_scores'] > opt.refine_thresh] *
                         torch.tensor([opt.input_w // 4, opt.input_h // 4], device=patch_area.device))[None]
    """if abandon_ind.sum() != 0:
        keep_ind = nms(boxes=torch.cat([ret['bboxes'][:, abandon_ind][0], refinement['bboxes'][0]]),
                       scores=torch.cat([torch.ones(abandon_ind.sum(), device=abandon_ind.device), refinement['scores']]),
                       iou_threshold=opt.nms_thre_nd)
        keep_ind = keep_ind[keep_ind >= abandon_ind.sum()]
        keep_ind = keep_ind if keep_ind.shape[0] == 0 else keep_ind - abandon_ind.sum()
        refinement['bboxes'] = refinement['bboxes'][:, keep_ind]
        refinement['scores'] = refinement['scores'][keep_ind]
        refinement['cts'] = refinement['cts'][:, keep_ind]"""
    refinement['bboxes_amodal'] = refinement['bboxes'].detach().clone()
    refinement['scores'] = refinement['scores'][None]
    #refinement['scores'] = torch.ones((1, refinement['bboxes'].shape[1]), dtype=torch.float32,
     #                                 device=ret['clses'].device)
    #refinement['scores'][refinement['scores'] > opt.refine_thresh][None] + 0.4 - opt.refine_thresh
    refinement['clses'] = torch.zeros((1, refinement['bboxes'].shape[1]), dtype=torch.float32,
                                      device=ret['clses'].device)
    #refinement['tracking'] = torch.zeros((1, refinement['bboxes'].shape[1], 2), dtype=torch.float32,
     #                                    device=ret['tracking'].device)
    #'''
    ret['tracking'] = torch.zeros((1, 100, 4), dtype=torch.float32,
                                      device=ret['clses'].device)
    for num in refinement:
        ret[num] = torch.cat((ret[num], refinement[num]), dim=1)

    if tes:
        return ret, num_pr, num_gt, num_gt_total
    else:
        return ret, ret_ori

def fake_track(output, patch_area, ind, mask):
    if output['refine_targets'][ind + 1]['boxes'].shape[0] != 0:
        gt = output['refine_targets'][ind + 1]['boxes'] # n_gt x 6
        motion_gt = output['refine_targets'][ind + 1]['motion']
    else:
        return {}, None, None

    tracker_ori = output['tracker_out']
    res_trans = torch.tensor([opt.ori_input_w / opt.input_w, opt.ori_input_h / opt.input_h], device=mask.device)
    patch_topx, patch_topy, patch_bottom_x, patch_bottom_y = (patch_area[ind] * res_trans.repeat(2)).int()
    patch_wh = torch.tensor([patch_bottom_x-patch_topx, patch_bottom_y-patch_topy], device=mask.device) # /4
    heat = tracker_ori['hm'][ind:(ind + 1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    batch, cat, height, width = heat.size()
    heat = _nms(heat)
    K = (heat != 0).sum() if (heat != 0).sum() <= opt.real_num_queries else torch.tensor(opt.real_num_queries,
                                                                                         device=patch_area.device)
    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)
    xs = xs0.view(batch, K, 1)
    ys = ys0.view(batch, K, 1)
    cts = torch.cat([xs.unsqueeze(2), ys.unsqueeze(2)], dim=2).squeeze(-1) # 1 x K x 2

    # get pre ct
    motion = tracker_ori['tracking'][ind:(ind + 1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    motion = _tranpose_and_gather_feat(motion, inds)  # B x K x 4
    motion = motion.view(batch, K, 2)
    pre_cts = cts.clone()# + motion # 1 x K x 2

    # heat [ind:(ind+1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    ltrb_amodal = tracker_ori['ltrb_amodal'][ind:(ind + 1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    ltrb_amodal = _tranpose_and_gather_feat(ltrb_amodal, inds)  # B x K x 4
    ltrb_amodal = ltrb_amodal.view(batch, K, 4)
    bboxes_amodal = torch.abs(torch.cat([ltrb_amodal[..., 0:1], ltrb_amodal[..., 1:2],
                                         ltrb_amodal[..., 2:3], ltrb_amodal[..., 3:4]], dim=2))

    fake_len = (gt.shape[0] * torch.clip((1 - opt.throw_baseline +
                              torch.randn([], device=gt.device) * opt.noise), min=0.1, max=1.)).ceil().int()
    keep = torch.ones(gt.shape[0]).bool().numpy()
    map = np.linspace(0, gt.shape[0]-1, gt.shape[0], dtype=np.long)
    propos = {}
    propos['cts'], propos['pre_cts'], propos['bboxes_amodal'], propos['pre_bboxes_amodal'] = [], [], [], []
    for fgt in range(fake_len): # fake_len already simulates the fn
        samp_ind = np.random.choice(map[keep], 1).squeeze() # sample a random absolute index for noise gt generation
        keep[samp_ind] = False
        gt_samp = gt[samp_ind].detach().clone() # 6
        gt_samp_pre = gt_samp.clone() # 6
        gt_samp_pre[:2] = gt_samp_pre[:2]# + motion_gt[samp_ind].detach().clone()

        gt1_cts, gt1_amodals = create_noisegt(gt_samp.clone())
        propos['cts'].append(gt1_cts)
        propos['bboxes_amodal'].append(gt1_amodals)
        gt1_pre_cts, gt1_pre_amodals = create_noisegt(gt_samp_pre.clone())
        propos['pre_cts'].append(gt1_pre_cts)
        propos['pre_bboxes_amodal'].append(gt1_pre_amodals)

        if torch.rand([]) < opt.duplicate_baseline: # simulate the fp with greater noise
            gt2_cts, gt2_amodals = create_noisegt(gt_samp.clone(), second=True)
            propos['cts'].append(gt2_cts)
            propos['bboxes_amodal'].append(gt2_amodals)
            gt2_pre_cts, gt2_pre_amodals = create_noisegt(gt_samp_pre.clone(), second=True)
            propos['pre_cts'].append(gt2_pre_cts)
            propos['pre_bboxes_amodal'].append(gt2_pre_amodals)

    propos['cts'] = torch.stack(propos['cts'])[None] * patch_wh # 1 x fake_len_act x 2
    propos['pre_cts'] = torch.stack(propos['pre_cts'])[None] * patch_wh
    propos['bboxes_amodal'] = torch.stack(propos['bboxes_amodal'])[None] * patch_wh.repeat(2) # 1 x fake_len_act x 4
    propos['pre_bboxes_amodal'] = torch.stack(propos['pre_bboxes_amodal'])[None] * patch_wh.repeat(2)
    fake_len_act = propos['cts'].shape[1]
    query_recon_mask = torch.linspace(0, opt.real_num_queries - 1, opt.real_num_queries).to(gt.device) < fake_len_act
    outbox = boxout2xyxy(torch.cat([propos['cts'], propos['bboxes_amodal']], dim=-1))[0] / patch_wh.repeat(2)
    iou = box_iou(outbox, boxout2xyxy(gt))  # n_pred x n_gt
    pred_gt_mat = generate_patch_gt(iou)  # real_num_queries   -1: no gt   >-1: has corresponding gt

    ite = torch.tensor(1 + (opt.real_num_queries - fake_len_act) / (scores < 0.4).sum(), device=gt.device).ceil().int()
    cts_s, bboxes_s = [], []
    cts_s_pre, bboxes_s_pre = [], []
    ct_random_amp = 0.03 if opt.eva else 0.05
    wh_random_amp = 0.05 if opt.eva else 0.1
    for i in range(ite):
        keep = torch.ones(fake_len_act, device=gt.device).bool()[None] if i == 0 else scores < 0.4  # 1 x K
        cts_ite = propos['cts'] if i == 0 else cts
        pre_cts_ite = propos['pre_cts'] if i == 0 else pre_cts
        amodal_ite = propos['bboxes_amodal'] if i == 0 else bboxes_amodal
        K = keep.sum()

        ct_noise = torch.randn(batch, K, 2).to(gt.device) * ct_random_amp
        retr = (ct_noise > 0).to(torch.long) * 2
        retr[..., 1] = retr[..., 1] + 1
        cts_ = cts_ite[keep][None].clone() + ct_noise * torch.gather(amodal_ite[keep][None], 2, retr)
        cts_s.append(cts_)
        # pre
        ct_noise_pre = torch.randn(batch, K, 2).to(gt.device) * ct_random_amp
        retr_pre = (ct_noise_pre > 0).to(torch.long) * 2
        retr_pre[..., 1] = retr_pre[..., 1] + 1
        pre_cts_ = pre_cts_ite[keep][None].clone() + ct_noise_pre * \
                  torch.gather(amodal_ite[keep][None], 2, retr_pre)
        cts_s_pre.append(pre_cts_)

        wh_nosie = torch.randn(batch, K, 2).to(gt.device) * wh_random_amp
        bboxes_amodal_ = amodal_ite[keep][None].clone() * (1 + wh_nosie.repeat(1, 1, 2))
        bboxes_s.append(bboxes_amodal_)
        # pre
        wh_nosie_pre = torch.randn(batch, K, 2).to(gt.device) * wh_random_amp
        bboxes_amodal_pre_ = amodal_ite[keep][None].clone() * (1 + wh_nosie_pre.repeat(1, 1, 2))
        bboxes_s_pre.append(bboxes_amodal_pre_)

    cts_s = torch.cat(cts_s, dim=1)[:, :opt.real_num_queries]
    bboxes_s = torch.cat(bboxes_s, dim=1)[:, :opt.real_num_queries]
    propos['cts'] = cts_s
    propos['bboxes_amodal'] = bboxes_s

    # pre
    cts_s_pre = torch.cat(cts_s_pre, dim=1)[:, :opt.real_num_queries]
    bboxes_s_pre = torch.cat(bboxes_s_pre, dim=1)[:, :opt.real_num_queries]
    propos['pre_cts'] = cts_s_pre
    propos['pre_bboxes_amodal'] = bboxes_s_pre

    return propos, query_recon_mask, pred_gt_mat

def create_noisegt(gt_samp, second=False):
    if not second:
        gt_samp[:2] = gt_samp[:2] + torch.randn(2, device=gt_samp.device) * opt.noise * (gt_samp[4:] + gt_samp[2:4]) / 2
        gt_samp[2:] = gt_samp[2:] * (1 + torch.randn(4, device=gt_samp.device) * opt.noise)
    else:
        gt_samp[:2] = gt_samp[:2] + torch.randn(2, device=gt_samp.device) * opt.noise * (gt_samp[4:] + gt_samp[2:4])
        gt_samp[2:] = gt_samp[2:] * (1 + torch.randn(4, device=gt_samp.device) * opt.noise * 2)
    return gt_samp[:2], gt_samp[2:]

def tracker_extract_disturb(output, patch_area, ind, mask): # batc, patch_area, patch_i, mask_pat
    tracker_ori = output['tracker_out']
    res_trans = torch.tensor([opt.ori_input_w / opt.input_w, opt.ori_input_h / opt.input_h], device=mask.device)
    patch_topx, patch_topy, patch_bottom_x, patch_bottom_y = (patch_area[ind] * res_trans.repeat(2)).int()
    patch_h, patch_w = patch_bottom_y - patch_topy, patch_bottom_x - patch_topx
    heat = tracker_ori['hm'][ind:(ind+1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    batch, cat, height, width = heat.size()
    #heat[mask[None] == 1] == 0
    heat = _nms(heat)
    K = (heat != 0).sum() if (heat != 0).sum() <= opt.real_num_queries else torch.tensor(opt.real_num_queries, device=patch_area.device)

    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)
    query_recon_mask = torch.linspace(0, opt.real_num_queries-1, opt.real_num_queries).to(heat.device) < (scores>0.4).sum()
    if not opt.eva: # to determine the validity of predictions whose score >=0.4
        ltrb_amodal_iou = tracker_ori['ltrb_amodal'][ind:(ind + 1), :, patch_topy:patch_bottom_y,
                          patch_topx:patch_bottom_x].detach().clone()
        ltrb_amodal_iou = _tranpose_and_gather_feat(ltrb_amodal_iou, inds)
        ltrb_amodal_iou = ltrb_amodal_iou.view(batch, K, 4)
        ltrb_amodal_iou = torch.cat([xs0.view(batch, K, 1) + ltrb_amodal_iou[..., 0:1],
                                     ys0.view(batch, K, 1) + ltrb_amodal_iou[..., 1:2],
                                     xs0.view(batch, K, 1) + ltrb_amodal_iou[..., 2:3],
                                     ys0.view(batch, K, 1) + ltrb_amodal_iou[..., 3:4]], dim=2)
        ltrb_amodal_iou = ltrb_amodal_iou / torch.tensor([patch_w, patch_h, patch_w, patch_h], device=ltrb_amodal_iou.device)
        if output['refine_targets'][ind + 1]['boxes'].shape[0] != 0:
            gt = boxout2xyxy(output['refine_targets'][ind + 1]['boxes']) # n_gt x 4
        else:
            return {}, None, None
        iou = box_iou(ltrb_amodal_iou[0, scores[0]>0.4], gt) # n_pred x n_gt
        pred_gt_mat = generate_patch_gt(iou) # real_num_queries   -1: no gt   >-1: has corresponding gt
    else:
        pred_gt_mat = torch.zeros(opt.real_num_queries, device=patch_area.device) - 1

    xs = xs0.view(batch, K, 1)
    ys = ys0.view(batch, K, 1)
    cts = torch.cat([xs.unsqueeze(2), ys.unsqueeze(2)], dim=2).squeeze(-1)
    propos = {'cts': cts}

    # get pre ct
    motion = tracker_ori['tracking'][ind:(ind + 1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    motion = _tranpose_and_gather_feat(motion, inds)  # B x K x 4
    motion = motion.view(batch, K, 2)
    pre_cts = cts.clone()# + torch.zeros_like(motion)
    propos['pre_cts'] = pre_cts

    # heat [ind:(ind+1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    ltrb_amodal = tracker_ori['ltrb_amodal'][ind:(ind+1), :, patch_topy:patch_bottom_y, patch_topx:patch_bottom_x]
    ltrb_amodal = _tranpose_and_gather_feat(ltrb_amodal, inds)  # B x K x 4
    ltrb_amodal = ltrb_amodal.view(batch, K, 4)
    bboxes_amodal = torch.abs(torch.cat([ltrb_amodal[..., 0:1], ltrb_amodal[..., 1:2],
                                         ltrb_amodal[..., 2:3], ltrb_amodal[..., 3:4]], dim=2))
    propos['bboxes_amodal'] = bboxes_amodal


    ite = torch.tensor(opt.real_num_queries / (K - query_recon_mask.sum()), device=xs.device).ceil().int()
    cts_s, bboxes_s = [], []
    cts_s_pre, bboxes_s_pre = [], []
    ct_random_amp = 0.03 if opt.eva else 0.05
    wh_random_amp = 0.05 if opt.eva else 0.1
    for i in range(ite):
        keep = scores > 0 if i == 0 else scores < 0.4 # 1 x K
        K = keep.sum()

        ct_noise = torch.randn(batch, K, 2).to(xs.device) * ct_random_amp
        retr = (ct_noise > 0).to(torch.long) * 2
        retr[..., 1] = retr[..., 1] + 1
        cts = propos['cts'][keep][None].clone() + ct_noise * \
              torch.gather(propos['bboxes_amodal'][keep][None], 2, retr)
        cts_s.append(cts)
        # pre
        ct_noise_pre = torch.randn(batch, K, 2).to(xs.device) * ct_random_amp
        retr_pre = (ct_noise_pre > 0).to(torch.long) * 2
        retr_pre[..., 1] = retr_pre[..., 1] + 1
        pre_cts = propos['pre_cts'][keep][None].clone() + ct_noise_pre * \
                  torch.gather(propos['bboxes_amodal'][keep][None], 2, retr_pre)
        cts_s_pre.append(pre_cts)

        wh_nosie = torch.randn(batch, K, 2).to(xs.device) * wh_random_amp
        bboxes_amodal = propos['bboxes_amodal'][keep][None].clone() * (1 + wh_nosie.repeat(1, 1, 2))
        bboxes_s.append(bboxes_amodal)
        # pre
        wh_nosie_pre = torch.randn(batch, K, 2).to(xs.device) * wh_random_amp
        bboxes_amodal_pre = propos['bboxes_amodal'][keep][None].clone() * (1 + wh_nosie_pre.repeat(1, 1, 2))
        bboxes_s_pre.append(bboxes_amodal_pre)

    cts_s = torch.cat(cts_s, dim=1)[:, :opt.real_num_queries]
    bboxes_s = torch.cat(bboxes_s, dim=1)[:, :opt.real_num_queries]
    propos['cts'] = cts_s
    propos['bboxes_amodal'] = bboxes_s

    # pre
    cts_s_pre = torch.cat(cts_s_pre, dim=1)[:, :opt.real_num_queries]
    bboxes_s_pre = torch.cat(bboxes_s_pre, dim=1)[:, :opt.real_num_queries]
    propos['pre_cts'] = cts_s_pre
    propos['pre_bboxes_amodal'] = bboxes_s_pre

    return propos, query_recon_mask, pred_gt_mat


def patch_tg_merg(indices, targets):
    """

    Args:
        indices:
        targets:

    Returns:

    """
    target_boxes = []
    for target, (pred_ind, tg_ind) in zip(targets, indices):
        if pred_ind.shape[0] == 0:
            continue
        target_xyxy = box_cxcywh_to_xyxy(target['boxes'])
        pred_ind_uni = torch.unique(pred_ind)
        for uni in range(pred_ind_uni.shape[0]):
            tgtbx = target_xyxy[tg_ind[pred_ind == pred_ind_uni[uni]]]
            tgtbx[0, :2] = tgtbx[:, :2].min(0).values
            tgtbx[0, 2:] = tgtbx[:, 2:].max(0).values
            target_boxes.append(tgtbx[0])

    return box_xyxy_to_cxcywh(torch.stack(target_boxes, dim=0))


def wh_img2patnorm(tgt_bbox, out_bbox, patch_area, sizes):
    """

    Args:
        tgt_bbox: n_gt x 4     center_x, center_y, w, h
        out_bbox: n x 4        center_x, center_y, w, h
        patch_area: psz x 4    topx, topy, botx, boty
        sizes: list of len psz indicating num of gt in each img

    Returns:

    """

    out_bbox_split = sizes if (tgt_bbox.shape[0] == out_bbox.shape[0]) else [opt.real_num_queries]*len(sizes)
    tgtbxs, outbxs = [], []
    for i, (tgtbx, outbx) in enumerate(zip(tgt_bbox.split(sizes, 0), out_bbox.split(out_bbox_split, 0))):
        tgtbx = tgtbx * ttbb211ss(patch_area[i:(i+1)]) if (not (0 in list(tgtbx.shape))) else torch.tensor([], device=tgt_bbox.device)
        outbx = outbx * ttbb211ss(patch_area[i:(i+1)]) if (not (0 in list(outbx.shape))) else torch.tensor([], device=tgt_bbox.device)
        tgtbxs.append(tgtbx)
        outbxs.append(outbx)

    return torch.cat(tgtbxs, dim=0), torch.cat(outbxs, dim=0)

def ttbb211ss(sub_patch_area):
    # sub_patch_area: 1 x 4
    sub_patch_area = torch.cat([torch.ones_like(sub_patch_area[:, :2], device=sub_patch_area.device),
                                sub_patch_area[:, 2:] - sub_patch_area[:, :2]], dim=-1)
    sub_patch_area[:, 2:] = 1. / sub_patch_area[:, 2:]

    return sub_patch_area

def boxout2xyxy(bbox):
    # bbox: cx, cy, amodtopx, amodtopy, amodbotx, amodboty    [..., 6]
    nnpp = torch.tensor([-1., -1., 1., 1.], device=bbox.device)
    ct, bbox_amodal = bbox[..., :2], bbox[..., 2:]
    bbox = bbox_amodal * nnpp + ct.repeat([1 if nu < (len(ct.shape) - 1) else 2 for nu in range(len(ct.shape))])
    return bbox

def affine_trans(ptmatrix, trans):
    # ptmatrix: bsz x 100 x 4 x 2    trans: bsz x 1 x 2 x 3
    ptmatrix = torch.cat([ptmatrix[..., None], torch.ones(ptmatrix.shape[:-1],
                             device=ptmatrix.device)[..., None, None]], dim=-2)  # bsz x 100 x 4 x 3 x 1

    return torch.matmul(trans[:, None].float().to(device=ptmatrix.device), ptmatrix).squeeze(-1).float()


def test_suspatch(opt, hm_down, oupu_wh, bac):

    fn_mas, fp_mas = mas_drawer(bac, hm_down, oupu_wh)
    patch_gt = torch.zeros(fn_mas.shape)
    for i_, pa in enumerate(bac['patch_targets']):
        out_hm = torch.zeros(fn_mas.shape[1:])
        for i in range(pa['boxes'].shape[0]):
            box = (torch.tensor([opt.input_w//4, opt.input_h//4, opt.input_w//4, opt.input_h//4], device=oupu_wh.device) *
                   pa['boxes'][i])
            box[:2] = box[:2] - box[2:] / 2
            box[2:] = box[:2] + box[2:]
            box = box.cpu().numpy()
            le = int(np.clip(box[0], 0, opt.input_w//4-1))
            to = int(np.clip(box[1], 0, opt.input_h//4-1))
            ri = int(np.clip(box[2], 0, opt.input_w//4-1))
            bo = int(np.clip(box[3], 0, opt.input_h//4-1))
            out_hm[0, to: (bo + 1), le: (ri + 1)] = np.maximum(out_hm[0, to: (bo + 1), le: (ri + 1)].cpu(), 1)
        patch_gt[i_] = out_hm

    patch_pred = torch.zeros(fn_mas.shape)
    for i_ in range(bac['patch_box_pred'].shape[0]):
        out_hm = torch.zeros(fn_mas.shape[1:])
        if (bac['patch_box_pred'][i_] - torch.tensor([0.5, 0.5, 1., 1.], device=oupu_wh.device)).sum() == 0:
            patch_pred[i_] = out_hm
            continue

        for i in range(opt.num_patch_val):
            # method 1
            #box = (torch.tensor([opt.input_w // 4, opt.input_h // 4, opt.input_w // 4, opt.input_h // 4], device=oupu_wh.device) *
             #      bac['patch_box_pred'][i_][
              #         torch.sort(bac['patch_box_cls'][i_, :, 1:].max(1).values, dim=0, descending=True).indices[i]]).squeeze()

            # method 2
            box = (torch.tensor([opt.input_w // 4, opt.input_h // 4, opt.input_w // 4, opt.input_h // 4],
                                device=oupu_wh.device) * bac['patch_area_pred'][i_]).squeeze()

            #box[:2] = box[:2] - box[2:] / 2
            #box[2:] = box[:2] + box[2:]
            box = box.detach().cpu().numpy()
            le = int(np.clip(box[0], 0, opt.input_w // 4 - 1))
            to = int(np.clip(box[1], 0, opt.input_h // 4 - 1))
            ri = int(np.clip(box[2], 0, opt.input_w // 4 - 1))
            bo = int(np.clip(box[3], 0, opt.input_h // 4 - 1))
            out_hm[0, to: (bo + 1), le: (ri + 1)] = np.maximum(out_hm[0, to: (bo + 1), le: (ri + 1)].cpu(), i+1)
            patch_pred[i_] = out_hm
        cts = bac['refer_p'][i_, :, :2] # bac['patch_area_pred'][i_, :2]
        cts = bac['patch_area_pred'][i_, :2] + (bac['patch_area_pred'][i_, 2:] - bac['patch_area_pred'][i_, :2]) * cts
        cts = (cts * torch.tensor([opt.input_w//4, opt.input_h//4], device=oupu_wh.device)).long().flip(-1)
        out_hm_referp = out_hm.clone().detach()
        out_hm_referp[0][cts[:, 0], cts[:, 1]] = 3

    bac['patch_area_pred_with_referp'] = out_hm_referp
    bac['patch_area_pred'] = patch_pred
    bac['patch_area'] = patch_gt


def draw_patch(opt, hm_down, oupu_wh, bac):

    fn_mas, fp_mas = mas_drawer(bac, hm_down, oupu_wh)
    patch = torch.zeros(fn_mas.shape, device=fn_mas.device)

    for i_ in range(fn_mas.shape[0]):
        ind_gt = torch.nonzero(fn_mas[i_].squeeze()).cpu().numpy()
        ind_hat = torch.nonzero(fp_mas[i_].squeeze()).cpu().numpy()
        out_hm = np.zeros((1, fp_mas.shape[2], fp_mas.shape[3]), dtype=np.float32)
        for m_ in range(ind_hat.shape[0]):
            w = oupu_wh[i_, 0, int(ind_hat[m_, 0]), int(ind_hat[m_, 1])].cpu().numpy()
            h = oupu_wh[i_, 1, int(ind_hat[m_, 0]), int(ind_hat[m_, 1])].cpu().numpy()
            le = int(np.maximum(ind_hat[m_, 1] - w / 2, 0))
            to = int(np.maximum(ind_hat[m_, 0] - h / 2, 0))
            ri = int(np.maximum(ind_hat[m_, 1] + w / 2, w-1))
            bo = int(np.maximum(ind_hat[m_, 0] + h / 2, h-1))
            out_hm[0, to: (bo + 1), le: (ri + 1)] = np.maximum(out_hm[0, to: (bo + 1), le: (ri + 1)], 1)

        for m_ in range(ind_gt.shape[0]):
            w = bac['hm_out_hw'][i_, 1, int(ind_gt[m_, 0]), int(ind_gt[m_, 1])].cpu().numpy()
            h = bac['hm_out_hw'][i_, 0, int(ind_gt[m_, 0]), int(ind_gt[m_, 1])].cpu().numpy()
            le = int(np.maximum(ind_gt[m_, 1] - w / 2, 0))
            to = int(np.maximum(ind_gt[m_, 0] - h / 2, 0))
            ri = int(np.maximum(ind_gt[m_, 1] + w / 2, w-1))
            bo = int(np.maximum(ind_gt[m_, 0] + h / 2, h-1))
            out_hm[0, to: (bo + 1), le: (ri + 1)] = np.maximum(out_hm[0, to: (bo + 1), le: (ri + 1)], 1)

        out_hm = torch.from_numpy(out_hm).to(fn_mas.device)
        patch[i_] = out_hm

    bac['patch_area'] = patch

    """
def draw_hm(opt, meta, out, down_=False):
    detector = Detector(opt)
    meta_copy = {}
    for nam in meta:
        meta_copy[nam] = meta[nam].detach().clone()
    if down_:
        hms_d = np.zeros((out[0]['hm'].shape[0], 1, opt.output_h, opt.output_w), dtype=np.float32)
    hms_ = np.zeros((out[0]['hm'].shape[0], 1, opt.input_h, opt.input_w), dtype=np.float32)
    dets_ = generic_decode(out[0], K=opt.K, opt=opt)
    for k in dets_:
        dets_[k] = dets_[k].detach().cpu().numpy()
    for na in meta_copy:
        meta_copy[na] = meta_copy[na].cpu().numpy()
    for b_ in range(out[0]['hm'].shape[0]):
        det, met = {}, {}
        for k in dets_:
            det[k] = dets_[k][b_][np.newaxis]
        for na in meta_copy:
            met[na] = meta_copy[na][b_]
        met['calib'] = None
        result = generic_post_process(
            opt, det, [met['c']], [met['s']],
            met['out_height'], met['out_width'], opt.num_classes,
            [met['calib']], met['height'], met['width'])
        detections = []
        detections.append(result[0])
        results = detector.merge_outputs(detections)
        hms__, _ = detector._get_additional_inputs(results, met)
        if down_:
            hms_d_, _ = detector._get_additional_inputs(results, met, down=down_)
            hms_d[b_] = hms_d_[0].cpu().numpy()
        hms_[b_] = hms__[0].cpu().numpy()
    if down_:
        return torch.from_numpy(hms_).to(out[0]['hm'].device), torch.from_numpy(hms_d).to(out[0]['hm'].device)
    return torch.from_numpy(hms_).to(out[0]['hm'].device)
    """

def draw_hm_mask(opt, mas, outpwh):

    out_hms = torch.zeros(mas.shape).to(outpwh.device)
    for i_ in range(mas.shape[0]):
        ind_ = torch.nonzero(mas[i_].squeeze()).cpu().numpy()
        out_hm = np.zeros((1, mas.shape[2], mas.shape[3]), dtype=np.float32)
        for m_ in range(ind_.shape[0]):
            w = outpwh[i_, 0, int(ind_[m_][0]), int(ind_[m_][1])].cpu().numpy()
            h = outpwh[i_, 1, int(ind_[m_][0]), int(ind_[m_][1])].cpu().numpy()
            le = int(np.maximum(ind_[m_][1] - w / 2, 0))
            to = int(np.maximum(ind_[m_][0] - h / 2, 0))
            ri = int(np.maximum(ind_[m_][1] + w / 2, w-1))
            bo = int(np.maximum(ind_[m_][0] + h / 2, h-1))
            out_hm[0][to: (bo + 1), le: (ri + 1)] = np.maximum(out_hm[0][to: (bo + 1), le: (ri + 1)], 1)
        out_hm = torch.from_numpy(out_hm).to(outpwh.device)
        out_hms[i_] = out_hm

    return out_hms

def hm_corr_genera(opt, hm_down, bac, oupu_wh):

    # lower fn
    bac['hm'][bac['hm'] == 1.0001] = -2
    fn_mas, fp_mas = mas_drawer(bac, hm_down, oupu_wh)
    #mostly_aligned_mas = (bac['hm'] * (hm_down == 1)) > 0.05
    #not_aligned_mas = ((hm_down * (bac['hm'] == 1)) <= 0.05) & (((hm_down + 1) * (bac['hm'] == 1)) >= 1)
    #fn_mas = mostly_aligned_mas | not_aligned_mas

    # lower fp
    #fp_mas = (((bac['hm'] + 1) * (hm_down == 1)) == 1)

    #fn_gt = torch.ones(fn_mas.shape).to(oupu_wh.device)
    #fp_gt = torch.zeros(fp_mas.shape).to(oupu_wh.device)
    #corr_gt = torch.ones(hm_down.shape).to(oupu_wh.device)

    #corr_gt[fn_mas == 1] = ((fn_gt + 1.) / (oupu[0]['hm'] + 1.))[fn_mas]
    #corr_gt[fp_mas == 1] = ((fp_gt + 1.) / (oupu[0]['hm'] + 1.))[fp_mas]

    # draw gt for neg loss training
    negloss_fn_gt = draw_negloss_gt(opt, fn_mas, oupu_wh)
    negloss_fp_gt = draw_negloss_gt(opt, fp_mas, oupu_wh)
    negloss_fn_gt[bac['hm'] == -2] = 1
    negloss_fp_gt[bac['hm'] == -2] = 1
    bac['negloss_fn_gt'] = negloss_fn_gt
    bac['fn_mask'] = fn_mas
    bac['negloss_fp_gt'] = negloss_fp_gt
    bac['fp_mask'] = fp_mas
    bac['hm'][bac['hm'] == -2] = 1

    return negloss_fn_gt, negloss_fp_gt

def mas_drawer(bach, hm_d, out_wh):

    fn_mases = torch.zeros(bach['hm'].shape).to(bach['hm'].device)
    fp_mases = torch.zeros(bach['hm'].shape).to(bach['hm'].device)
    for i_ in range(hm_d.shape[0]):
        ind_gt = torch.nonzero((bach['hm'][i_] == 1).squeeze())  # nx2
        ind_hat = torch.nonzero((hm_d[i_] == 1).squeeze())  # mx2
        num_gt, num_hat = ind_gt.shape[0], ind_hat.shape[0]
        dist = (torch.abs(ind_gt.reshape(-1, 1, 2) - ind_hat.reshape(1, -1, 2)).sum(2)) # nxm
        matched_coors = []
        unmatched_coors = []

        if min(num_gt, num_hat) == 0:
          if num_gt == 0:
            for hat_i in range(num_hat):
              unmatched_coors.append(ind_hat[hat_i, :])
          else:
            for gt_i in range(num_gt):
              matched_coors.append(ind_gt[gt_i, :])
        else:
          row_ind, col_ind = linear_sum_assignment(dist.cpu().numpy())
          for i in range(len(row_ind)):
            if dist[row_ind[i], col_ind[i]] >= 15:
              matched_coors.append(ind_gt[row_ind[i], :])
              unmatched_coors.append(ind_hat[col_ind[i], :])
          if num_gt > num_hat:
            for gt_i in range(num_gt):
              if not (gt_i in row_ind):
                matched_coors.append(ind_gt[gt_i, :])
          elif num_gt < num_hat:
            for hat_i in range(num_hat):
              if not (hat_i in col_ind):
                unmatched_coors.append(ind_hat[hat_i, :])

        for m_ in range(len(matched_coors)):
            fn_mases[i_, 0, int(matched_coors[m_][0]), int(matched_coors[m_][1])] = 1
        for n_ in range(len(unmatched_coors)):
            fp_mases[i_, 0, int(unmatched_coors[n_][0]), int(unmatched_coors[n_][1])] = 1

    return (fn_mases == 1), (fp_mases == 1)

def draw_negloss_gt(opt, mas, outp):

    out_hms = torch.zeros(mas.shape).to(mas.device)
    for i_ in range(mas.shape[0]):
        ind_ = torch.nonzero(mas[i_].squeeze())
        out_hm = np.zeros((1, mas.shape[2], mas.shape[3]), dtype=np.float32)
        for m_ in range(ind_.shape[0]):
            w = outp[i_, 0, int(ind_[m_][0]), int(ind_[m_][1])]
            h = outp[i_, 1, int(ind_[m_][0]), int(ind_[m_][1])]
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            ct_int = np.flip(ind_[m_].cpu().numpy())
            draw_umich_gaussian(out_hm[0], ct_int, int(radius+1))
        out_hm = torch.from_numpy(out_hm).to(outp.device)
        out_hms[i_] = out_hm

    return out_hms

"""
from post_transfiner.transfiner import build
from dataset.dataset_factory import get_dataset
class Detector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(
            opt.arch, opt.heads, opt.head_conv, opt=opt)
        self.model_second, _, _ = build(opt)
        self.model, self.model_second = load_model(self.model, opt.load_model, self.model_second, opt)
        self.model, self.model_second = \
            self.model.to(opt.device), self.model_second.to(opt.device)
        self.model.eval()
        self.model_second.eval()
        # self.model_fusion.eval()

        self.opt = opt
        self.trained_dataset = get_dataset(opt.dataset)
        self.mean = np.array(
            self.trained_dataset.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
        self.pause = not opt.no_pause
        self.rest_focal_length = self.trained_dataset.rest_focal_length \
            if self.opt.test_focal_length < 0 else self.opt.test_focal_length
        self.flip_idx = self.trained_dataset.flip_idx
        self.cnt = 0
        self.pre_images = None
        self.pre_image_ori = None
        self.tracker = Tracker(opt)
        self.tracker_temp = Tracker(opt)
        self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)

    def run(self, image_or_path_or_tensor, meta={}):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
        self.debugger.clear()
        start_time = time.time()

        # read image
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []

        # for multi-scale testing
        for scale in self.opt.test_scales:
            scale_start_time = time.time()
            if not pre_processed:
                # not prefetch testing or demo
                images, meta = self.pre_process(image, scale, meta)
            else:
                # prefetch testing
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
                if 'pre_dets' in pre_processed_images['meta']:
                    meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
                if 'cur_dets' in pre_processed_images['meta']:
                    meta['cur_dets'] = pre_processed_images['meta']['cur_dets']

            images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)

            # initializing tracker
            pre_hms, hms, pre_inds = None, None, None
            if self.opt.tracking:
                # initialize the first frame
                if self.pre_images is None:
                    print('Initialize tracking!')
                    self.pre_images = images
                    self.tracker.init_track(
                        meta['pre_dets'] if 'pre_dets' in meta else [])
                    self.tracker_temp.init_track(
                        meta['pre_dets'] if 'pre_dets' in meta else [])
                if self.opt.pre_hm:
                    # render input heatmap from tracker status
                    # pre_inds is not used in the current version.
                    # We used pre_inds for learning an offset from previous image to
                    # the current image.
                    pre_hms, pre_inds = self._get_additional_inputs(
                        self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)

            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            # run the network
            # output: the output feature maps, only used for visualizing
            # dets: output tensors after extracting peaks
            output, dets, forward_time = self.process(
                images, self.pre_images, pre_hms, pre_inds=pre_inds, return_time=True)
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            # convert the cropped and 4x downsampled output coordinate system
            # back to the input image coordinate system
            result = self.post_process(dets, meta, scale)
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(result)
            if self.opt.debug >= 2:
                self.debug(
                    self.debugger, images, result, output, scale,
                    pre_images=self.pre_images if not self.opt.no_pre_img else None,
                    pre_hms=pre_hms)

        # merge multi-scale testing results
        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time

        if self.opt.tracking:
            # public detection mode in MOT challenge
            public_det = meta['cur_dets'] if self.opt.public_det else None
            # add tracking id to results
            _ = self.tracker_temp.step(results, public_det)
            hms, _ = self._get_additional_inputs(
                self.tracker_temp.tracks, meta, with_hm=not self.opt.zero_pre_hm)
            output, dets, forward_time = self.process(
                images, self.pre_images, pre_hms, hms, pre_inds, return_time=True)
            result = self.post_process(dets, meta, scale)
            detections = []
            detections.append(result)
            results = self.merge_outputs(detections)
            results = self.tracker.step(results, public_det)
            self.pre_images = images

        tracking_time = time.time()
        track_time += tracking_time - end_time
        tot_time += tracking_time - start_time

        if self.opt.debug >= 1:
            self.show_results(self.debugger, image, results)
        self.cnt += 1

        show_results_time = time.time()
        display_time += show_results_time - end_time

        # return results and run time
        ret = {'results': results, 'tot': tot_time, 'load': load_time,
               'pre': pre_time, 'net': net_time, 'dec': dec_time,
               'post': post_time, 'merge': merge_time, 'track': track_time,
               'display': display_time}
        if self.opt.save_video:
            try:
                # return debug image for saving video
                ret.update({'generic': self.debugger.imgs['generic']})
            except:
                pass
        return ret

    def _transform_scale(self, image, scale=1):
        '''
          Prepare input image in different testing modes.
            Currently support: fix short size/ center crop to a fixed size/
            keep original resolution but pad to a multiplication of 32
        '''
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_short > 0:
            if height < width:
                inp_height = self.opt.fix_short
                inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
            else:
                inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
                inp_width = self.opt.fix_short
            c = np.array([width / 2, height / 2], dtype=np.float32)
            s = np.array([width, height], dtype=np.float32)
        elif self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
            # s = np.array([inp_width, inp_height], dtype=np.float32)
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, c, s, inp_width, inp_height, height, width

    def pre_process(self, image, scale, input_meta={}):
        '''
        Crop, resize, and normalize image. Gather meta data for post processing
          and tracking.
        '''
        resized_image, c, s, inp_width, inp_height, height, width = \
            self._transform_scale(image)
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        out_height = inp_height // self.opt.down_ratio
        out_width = inp_width // self.opt.down_ratio
        trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
            if 'calib' in input_meta else \
            self._get_default_calib(width, height)}
        meta.update({'c': c, 's': s, 'height': height, 'width': width,
                     'out_height': out_height, 'out_width': out_width,
                     'inp_height': inp_height, 'inp_width': inp_width,
                     'trans_input': trans_input, 'trans_output': trans_output})
        if 'pre_dets' in input_meta:
            meta['pre_dets'] = input_meta['pre_dets']
        if 'cur_dets' in input_meta:
            meta['cur_dets'] = input_meta['cur_dets']
        return images, meta

    def _trans_bbox(self, bbox, trans, width, height):
        '''
        Transform bounding boxes according to image crop.
        '''
        bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
        bbox[:2] = affine_transform(bbox[:2], trans)
        bbox[2:] = affine_transform(bbox[2:], trans)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        return bbox

    def _get_additional_inputs(self, dets, meta, with_hm=True, down=False):
        '''
        Render input heatmap from previous trackings.
        '''
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']
        if down:
            input_hm = np.zeros((1, out_height, out_width), dtype=np.float32)
        else:
            input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)
        output_inds = []
        for det in dets:
            if det['score'] < self.opt.pre_thresh or ((det['active'] == 0) if 'active' in det else False):
                continue
            bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
            bbox_out = self._trans_bbox(
                det['bbox'], trans_output, out_width, out_height)
            if down:
                h, w = bbox_out[3] - bbox_out[1], bbox_out[2] - bbox_out[0]
            else:
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                if down:
                    ct = np.array(
                        [(bbox_out[0] + bbox_out[2]) / 2, (bbox_out[1] + bbox_out[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                else:
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                if with_hm:
                    draw_umich_gaussian(input_hm[0], ct_int, radius)
                ct_out = np.array(
                    [(bbox_out[0] + bbox_out[2]) / 2,
                     (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
                output_inds.append(ct_out[1] * out_width + ct_out[0])
        if with_hm:
            input_hm = input_hm[np.newaxis]
            if self.opt.flip_test:
                input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
            input_hm = torch.from_numpy(input_hm).to(self.opt.device)
        output_inds = np.array(output_inds, np.int64).reshape(1, -1)
        output_inds = torch.from_numpy(output_inds).to(self.opt.device)
        return input_hm, output_inds

    def _get_default_calib(self, width, height):
        calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                          [0, self.rest_focal_length, height / 2, 0],
                          [0, 0, 1, 0]])
        return calib

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = output['hm'].sigmoid_()
        if 'hm_hp' in output:
            output['hm_hp'] = output['hm_hp'].sigmoid_()
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            output['dep'] *= self.opt.depth_scale
        return output

    def _flip_output(self, output):
        average_flips = ['hm', 'wh', 'dep', 'dim']
        neg_average_flips = ['amodel_offset']
        single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
                        'hp_offset', 'rot', 'tracking', 'pre_hm']
        for head in output:
            if head in average_flips:
                output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
            if head in neg_average_flips:
                flipped_tensor = flip_tensor(output[head][1:2])
                flipped_tensor[:, 0::2] *= -1
                output[head] = (output[head][0:1] + flipped_tensor) / 2
            if head in single_flips:
                output[head] = output[head][0:1]
            if head == 'hps':
                output['hps'] = (output['hps'][0:1] +
                                 flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
            if head == 'hm_hp':
                output['hm_hp'] = (output['hm_hp'][0:1] + \
                                   flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2

        return output

    def process(self, images, pre_images=None, pre_hms=None, hms=None,
                pre_inds=None, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            with autocast(enabled=self.opt.withamp):

                output, feats = self.model(images, pre_img=pre_images, pre_hm=pre_hms)
                output[0] = self._sigmoid_output(output[0])
                # inputs_second = second_round(self.opt, output)
                # hm_corr_hat, hm_pr_hat = self.model_second(inputs_second.to(output[0]['hm'].device))
                if hms != None:
                    inputs_second = self.second_round(self.opt, output, hms)
                    hm_corr_hat, hm_pr_hat, _ = self.model_second(inputs_second)
                    hm_corr_hat = self.hmc((hm_corr_hat - hm_pr_hat), self.opt.divide)
                    pre_hms = (output[0]['hm'] + 1.) * hm_corr_hat - 1.

                    # addition
                    output[0]['hm'] = pre_hms
                    # output[0]['hm'] = (output[0]['hm'] + 1.) * hm_corr_hat - 1.

            output[0].update({'pre_inds': pre_inds})
            if self.opt.flip_test:
                output[0] = self._flip_output(output[0])
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = generic_decode(output[0], K=self.opt.K, opt=self.opt)
            torch.cuda.synchronize()
            for k in dets:
                dets[k] = dets[k].detach().cpu().numpy()
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = generic_post_process(
            self.opt, dets, [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes,
            [meta['calib']], meta['height'], meta['width'])
        self.this_calib = meta['calib']

        if scale != 1:
            for i in range(len(dets[0])):
                for k in ['bbox', 'hps']:
                    if k in dets[0][i]:
                        dets[0][i][k] = (np.array(
                            dets[0][i][k], np.float32) / scale).tolist()
        return dets[0]

    def merge_outputs(self, detections):
        assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
        results = []
        for i in range(len(detections[0])):
            if detections[0][i]['score'] > self.opt.out_thresh:
                results.append(detections[0][i])
        return results

    def debug(self, debugger, images, dets, output, scale=1,
              pre_images=None, pre_hms=None):
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if 'hm_hp' in output:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

        if pre_images is not None:
            pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            pre_img = np.clip(((
                                       pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
            debugger.add_img(pre_img, 'pre_img')
            if pre_hms is not None:
                pre_hm = debugger.gen_colormap(
                    pre_hms[0].detach().cpu().numpy())
                debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='generic')
        if self.opt.tracking:
            debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image,
                             img_id='previous')
            self.pre_image_ori = image

        for j in range(len(results)):
            if results[j]['score'] > self.opt.vis_thresh:
                if 'active' in results[j] and results[j]['active'] == 0:
                    continue
                item = results[j]
                if ('bbox' in item):
                    sc = item['score'] if self.opt.demo == '' or \
                                          not ('tracking_id' in item) else item['tracking_id']
                    sc = item['tracking_id'] if self.opt.show_track_color else sc

                    debugger.add_coco_bbox(
                        item['bbox'], item['class'] - 1, sc, img_id='generic')

                if 'tracking' in item:
                    debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')

                tracking_id = item['tracking_id'] if 'tracking_id' in item else -1
                if 'tracking_id' in item and self.opt.demo == '' and \
                        not self.opt.show_track_color:
                    debugger.add_tracking_id(
                        item['ct'], item['tracking_id'], img_id='generic')

                if (item['class'] in [1, 2]) and 'hps' in item:
                    debugger.add_coco_hp(item['hps'], tracking_id=tracking_id,
                                         img_id='generic')

        if len(results) > 0 and \
                'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
            debugger.add_3d_detection(
                image if not self.opt.qualitative else cv2.resize(
                    debugger.imgs['pred_hm'], (image.shape[1], image.shape[0])),
                False, results, self.this_calib,
                vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
            debugger.add_bird_view(
                results, vis_thresh=self.opt.vis_thresh,
                img_id='bird_pred', cnt=self.cnt)
            if self.opt.show_track_color and self.opt.debug == 4:
                del debugger.imgs['generic'], debugger.imgs['bird_pred']
        if 'ddd_pred' in debugger.imgs:
            debugger.imgs['generic'] = debugger.imgs['ddd_pred']
        if self.opt.debug == 4:
            debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))
        else:
            debugger.show_all_imgs(pause=self.pause)

    def reset_tracking(self):
        self.tracker.reset()
        self.pre_images = None
        self.pre_image_ori = None
    """
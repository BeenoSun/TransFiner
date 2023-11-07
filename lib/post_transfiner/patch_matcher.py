# Modified by Peize Sun, Rufeng Zhang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from post_transfiner.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_cxcyhw_to_xyxy, ciou, box_xyxy_to_cxcywh
import numpy as np
from opts import opts
opt = opts().parse()

class PatchMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, patch_cls, patch_box, bachs):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_patch_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_patch_queries, 4] with the predicted box coordinates logits
                               in the form of 'center_x, center_y, w, h'

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

            bachs: A dict contains helpful information to determine the cls & box coords of patchs, including:
                 "patch_area": Tensor of dim [batch_size, 1, output_height, output_width]
                 "fn_mask": Tensor of dim [batch_size, 1, output_height, output_width]
                 "fp_mask": Tensor of dim [batch_size, 1, output_height, output_width]
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():

            bs, num_queries = patch_box.shape[:2]
            # We flatten to compute the cost matrices in a batch
            patch_box, patch_cls = patch_box.flatten(0, 1), patch_cls.flatten(0, 1).sigmoid()

            # Also concat the target labels and boxes
            #tgt_ids = torch.cat([v["labels"] for v in bachs['patch_targets']]).long() # n_gt
            if torch.cat([v["labels"] for v in bachs['patch_targets']]).shape[0] == 0:
                return [], list(np.arange(0, bs, 1, dtype=np.int)), torch.tensor([], device=patch_box.device)
            tgt_ids = torch.stack([torch.tensor(len(v["cls"])) for v in bachs['patch_targets'] if v["cls"].shape[0] != 0]).long()
            #tgt_cls = torch.cat([v["cls"] for v in bachs['patch_targets']]) # identifier for fp(false positive) or fn(false negative)
            # bachs['patch_targets']["boxes"]: 'center_x, center_y, w, h'
            tgt_bbox = torch.cat([v["boxes"] for v in bachs['patch_targets']]).float().to(patch_box.device) # n_gt x 4

            # Compute the classification cost.
            alpha = opt.patch_focal_alpha
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (patch_cls ** gamma) * (-(1 - patch_cls + 1e-8).log())
            pos_cost_class = alpha * ((1 - patch_cls) ** gamma) * (
                -(patch_cls + 1e-8).log())  # pos&neg balance x easy&hard balance x cls loss
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]  # (bsz x n_q) x n_gt

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(patch_box, tgt_bbox, p=1)  # (bsz x n_q) x n_gt

            # Compute the giou cost betwen boxes
            #try:
            #cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(patch_box), box_cxcywh_to_xyxy(tgt_bbox))
            cost_giou = -ciou(patch_box, tgt_bbox)
            #except:
             #   print(1)
            #wh_r = (patch_box[:, 2:3] + 1e-5) / (patch_box[:, 3:4] + 1e-5) # (bsz x n_q) x 1
            #wh_r_cost = (wh_r - 1.45).relu() + (-wh_r + 0.55).relu()

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou# + opt.whr_norm_coef * wh_r_cost
            C = C.view(bs, num_queries, -1).cpu()

            # split the matrix to have the matching indexes for each img
            sizes = [len(v["boxes"]) for v in bachs['patch_targets']]
            indices = [patch_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] # c[i]: n_p_q x n_gt
            indices_tg = [patch_assignment(c[i], tg=True) for i, c in enumerate(C.split(sizes, -1))]
            indices_tg = [(torch.as_tensor(i, dtype=torch.int64, device=patch_box.device),
                           torch.as_tensor(j, dtype=torch.int64, device=patch_box.device)) for i, j in indices_tg]

            # produce a patch to batch match list for the case there are multiple patches for a single img
            sizes = [len(v[0]) if len(v[0]) != 0 else 1 for v in indices]
            patch2batch = [[i] * c if c != 0 else [i] for i, c in enumerate(sizes)]
            patch2batch = [item for sublist in patch2batch for item in sublist]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], patch2batch, indices_tg

def patch_assignment(cost_mat, tg=False):

    if cost_mat.shape[1] == 0:
        return (np.array([], dtype=np.int), np.array([], dtype=np.int))
    else:
        _, row_ind = torch.min(cost_mat, dim=0)
        row_ind = np.array(list(set(row_ind.numpy()))) if not tg else row_ind.numpy()
        col_ind = np.arange(0, cost_mat.shape[1], 1, dtype=np.int)

    return (row_ind, col_ind)

def build_patch_matcher(args):
    return PatchMatcher(cost_class=args.set_p_cost_class,
                            cost_bbox=args.set_p_cost_bbox,
                            cost_giou=args.set_p_cost_giou)

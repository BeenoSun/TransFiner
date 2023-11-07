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
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from post_transfiner.utils.utils import wh_img2patnorm, patch_tg_merg, boxout2xyxy, pre2samp, amodal2points, patchfilter
from post_transfiner.utils import box_ops
from post_transfiner.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from post_transfiner.model_.detector import build_patchdet
from post_transfiner.model_.refine_backbone import build_backbone
from post_transfiner.model_.tnt import build_transpatch_tnt
from post_transfiner.model_.pyramid_tnt import build_transpatch_ptnt
from post_transfiner.matcher import build_matcher
from post_transfiner.patch_matcher import build_patch_matcher
from post_transfiner.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from torchvision.ops import nms
from .deformable_transformer import build_deforamble_transformer
from .deformable_transformer_eval import build_deforamble_transformer_eval
import copy
from post_transfiner.patch_matcher import PatchMatcher
from torch.cuda.amp import autocast, GradScaler
from scipy.optimize import linear_sum_assignment
from opts import opts
opt = opts().parse()

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transfiner(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer

        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4 if not opt.box_amodal_pred else 6, 3)
        self.motion_embed = MLP(hidden_dim, hidden_dim, 4 if not opt.box_amodal_pred else 6, 3)
        self.num_feature_levels = num_feature_levels
        self.patchmatcher = PatchMatcher()
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1)])
        self.combine = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

        self.backbone = backbone
        #self.transpatch = transpatch
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.postprocess = PostProcess()
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.motion_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.motion_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        if not opt.transpatch_trainonly:
            # if two-stage, the last class_embed and bbox_embed is for region proposal generation
            num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
            if with_box_refine:
                self.class_embed = _get_clones(self.class_embed, num_pred)
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
                self.motion_embed = _get_clones(self.motion_embed, num_pred)
                nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
                nn.init.constant_(self.motion_embed[0].layers[-1].bias.data[2:], -2.0)
                # hack implementation for iterative bounding box refinement
                self.transformer.decoder.bbox_embed = self.bbox_embed
                self.transformer.decoder.motion_embed = self.motion_embed
            else:
                nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
                self.transformer.decoder.bbox_embed = None
            if two_stage:
                # hack implementation for two-stage
                self.transformer.decoder.class_embed = self.class_embed
                for box_embed in self.bbox_embed:
                    nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    @torch.no_grad()
    def randshift(self, samples, targets):
        bs = samples.tensors.shape[0]

        self.xshift = (100 * torch.rand(bs)).int()
        self.xshift *= (torch.randn(bs) > 0.0).int() * 2 - 1
        self.yshift = (100 * torch.rand(bs)).int()
        self.yshift *= (torch.randn(bs) > 0.0).int() * 2 - 1

        shifted_images = []
        new_targets = copy.deepcopy(targets)

        for i, (image, target) in enumerate(zip(samples.tensors, targets)):
            _, h, w = image.shape
            img_h, img_w = target['size']
            nopad_image = image[:, :img_h, :img_w]
            image_patch = \
                nopad_image[:,
                max(0, -self.yshift[i]): min(h, h - self.yshift[i]),
                max(0, -self.xshift[i]): min(w, w - self.xshift[i])]

            _, patch_h, patch_w = image_patch.shape
            ratio_h, ratio_w = img_h / patch_h, img_w / patch_w
            shifted_image = F.interpolate(image_patch[None], size=(img_h, img_w))[0]
            pad_shifted_image = copy.deepcopy(image)
            pad_shifted_image[:, :img_h, :img_w] = shifted_image
            shifted_images.append(pad_shifted_image)

            scale = torch.tensor([img_w, img_h, img_w, img_h], device=image.device)[None]
            bboxes = target['boxes'] * scale
            bboxes -= torch.tensor([max(0, -self.xshift[i]), max(0, -self.yshift[i]), 0, 0], device=image.device)[None]
            bboxes *= torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h], device=image.device)[None]
            shifted_bboxes = bboxes / scale
            new_targets[i]['boxes'] = shifted_bboxes

        new_samples = copy.deepcopy(samples)
        new_samples.tensors = torch.stack(shifted_images, dim=0)

        return new_samples, new_targets

    def forward(self, samples_targets):
        if self.training or not opt.eva:
            samples, pre_samples, targets = samples_targets
            out = self.forward_train(samples, pre_samples, targets)

            return out

        else:
            samples, pre_samples, targets = samples_targets
            out = self.forward_eval(samples, pre_samples, targets)
            result_refine, patch_bbox = self.postprocess(out)
            return result_refine, patch_bbox
    #@torch.no_grad()
    def forward_train(self, samples: NestedTensor, pre_samples: NestedTensor, bch):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        with torch.no_grad():
            pre_features, pre_pos = self.backbone(pre_samples)

        srcs = []
        masks = []
        pre_srcs = []
        pre_masks = []
        for l, (feat, pre_feat) in enumerate(zip(features, pre_features)):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)

            pre_src, pre_mask = pre_feat.decompose()
            pre_srcs.append(self.input_proj[l](pre_src))
            pre_masks.append(pre_mask)
            assert mask is not None
            assert pre_mask is not None
            assert pre_src.shape == src.shape

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                    pre_src = self.input_proj[l](pre_features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                    pre_src = self.input_proj[l](pre_srcs[-1])
                assert pre_src.shape == src.shape
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

                # pre
                pre_m = pre_samples.mask
                pre_mask = F.interpolate(pre_m[None].float(), size=pre_src.shape[-2:]).to(torch.bool)[0]
                pre_pos_l = self.backbone[1](NestedTensor(pre_src, pre_mask)).to(src.dtype)
                pre_srcs.append(pre_src)
                pre_masks.append(pre_mask)
                pre_pos.append(pre_pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, pre_hs, _, gt_patch_matched, query_recon_masks, pred_gt_mats = \
            self.transformer(srcs, masks, pos, query_embeds, bch, pre_srcs=pre_srcs,
                             pre_masks=pre_masks, pre_hms=None, pre_pos_embeds=pre_pos)
        #hs, init_reference_out, inter_references_out, pre_hs, patch_area, src_valid_trans, query_recon_masks, pred_gt_mats
        outputs_classes = []
        outputs_coords = []
        outputs_coords_pre = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])

            pre_tmp = self.motion_embed[lvl](pre_hs[lvl])
            if reference.shape[-1] == (4 if not opt.box_amodal_pred else 6):
                tmp += reference
                pre_tmp = reference + pre_tmp
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            outputs_coord_pre = pre_tmp.sigmoid()
            outputs_coords_pre.append(outputs_coord_pre)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_coords_pre = torch.stack(outputs_coords_pre)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pre_boxes': outputs_coords_pre[-1],
               'query_recon_masks': query_recon_masks, 'pred_gt_mats': pred_gt_mats}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_coords_pre)

        if self.two_stage and not opt.transpatch_trainonly:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coords_pre=None):

        if outputs_coords_pre is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pre_boxes': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_coords_pre[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @torch.no_grad()
    @autocast(enabled=opt.withamp)
    def forward_eval(self, samples: NestedTensor, pre_samples: NestedTensor, bch):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        with torch.no_grad():
            pre_features, pre_pos = self.backbone(pre_samples)
        srcs = []
        masks = []
        pre_srcs = []
        pre_masks = []
        for l, (feat, pre_feat) in enumerate(zip(features, pre_features)):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)

            pre_src, pre_mask = pre_feat.decompose()
            pre_srcs.append(self.input_proj[l](pre_src))
            pre_masks.append(pre_mask)
            assert mask is not None
            assert pre_mask is not None
            assert pre_src.shape == src.shape

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                    pre_src = self.input_proj[l](pre_features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                    pre_src = self.input_proj[l](pre_srcs[-1])
                assert pre_src.shape == src.shape
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

                # pre
                pre_m = pre_samples.mask
                pre_mask = F.interpolate(pre_m[None].float(), size=pre_src.shape[-2:]).to(torch.bool)[0]
                pre_pos_l = self.backbone[1](NestedTensor(pre_src, pre_mask)).to(src.dtype)
                pre_srcs.append(pre_src)
                pre_masks.append(pre_mask)
                pre_pos.append(pre_pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, pre_hs, patch_area, src_valid_trans, _, _ = self.transformer(
            srcs, masks, pos, query_embeds, bch,
            pre_srcs=pre_srcs, pre_masks=pre_masks, pre_hms=None, pre_pos_embeds=pre_pos)

        outputs_classes = []
        outputs_coords = []
        outputs_coords_pre = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])

            pre_tmp = self.motion_embed[lvl](pre_hs[lvl])
            if reference.shape[-1] == (4 if not opt.box_amodal_pred else 6):
                tmp += reference
                pre_tmp = reference + pre_tmp
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            outputs_coord_pre = pre_tmp.sigmoid()
            outputs_coords_pre.append(outputs_coord_pre)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_coords_pre = torch.stack(outputs_coords_pre)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'pre_boxes': outputs_coords_pre[-1], 'src_valid_trans': src_valid_trans,
               'patch_area': patch_area}

        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, patch_matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            patch_matcher:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.patch_matcher = patch_matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # bsz x n_q x n_cls
        if num_boxes == 0:
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        else:
            idx = self._get_src_permutation_idx(indices)
            #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes_o = torch.ones(idx[0].shape[0], dtype=torch.int64, device=targets[0]['patch_area'].device)
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes if num_boxes != 0 else 1.,
                                     alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        #if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            #losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        losses = {}
        if num_boxes == 0:
            losses['loss_bbox'] = torch.zeros([], device=outputs['pred_boxes'].device)
            losses['loss_giou'] = torch.zeros([], device=outputs['pred_boxes'].device)
            losses['pre_loss_bbox'] = torch.zeros([], device=outputs['pred_boxes'].device)
            losses['pre_loss_giou'] = torch.zeros([], device=outputs['pred_boxes'].device)
            return losses

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets[1:], indices) if 'boxes' in t], dim=0)
        target_motions = torch.cat([t['motion'][i] for t, (_, i) in zip(targets[1:], indices) if 'motion' in t], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        sizes = [len(v["boxes"]) for v in targets if 'boxes' in v]
        normed_out_bx, normed_tgt_bx = boxout2xyxy(src_boxes), boxout2xyxy(target_boxes)
        loss_giou = 1 - torch.diag(box_ops.ciou(box_ops.box_xyxy_to_cxcywh(normed_out_bx), box_ops.box_xyxy_to_cxcywh(normed_tgt_bx)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # pre loss
        pre_src_boxes = outputs['pre_boxes'][idx]
        #target_boxes_pre = target_boxes.detach().clone()
        target_boxes_pre = target_motions

        loss_bbox_pre = F.l1_loss(pre_src_boxes, target_boxes_pre, reduction='none')
        losses['pre_loss_bbox'] = loss_bbox_pre.sum() / num_boxes
        normed_out_bx_pre, normed_tgt_bx_pre = boxout2xyxy(pre_src_boxes), boxout2xyxy(target_boxes_pre)
        loss_giou_pre = 1 - torch.diag(
            box_ops.ciou(box_ops.box_xyxy_to_cxcywh(normed_out_bx_pre), box_ops.box_xyxy_to_cxcywh(normed_tgt_bx_pre)))
        losses['pre_loss_giou'] = loss_giou_pre.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def patch_loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        if num_boxes == 0:
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        else:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = [len(t["cls"]) for t, (_, J) in zip(targets, indices) if J.shape[0] != 0]
            target_classes_o = torch.tensor(target_classes_o, dtype=torch.int64, device=targets[0]['boxes'].device)
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes if num_boxes != 0 else 1.,
                                     alpha=opt.patch_focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'patch_loss_ce': loss_ce}

        #if log:
            # TODO this should probably be a separate loss, not hacked in this one here
         #   losses['patch_loss_ce'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def patch_loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        losses = {}
        if num_boxes == 0:
            losses['patch_loss_bbox'] = torch.zeros([], device=outputs['pred_boxes'].device)
            losses['patch_loss_giou'] = torch.zeros([], device=outputs['pred_boxes'].device)
            return losses

        # patch tg merging
        #target_boxes = patch_tg_merg(indices, targets)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices) if 'boxes' in t], dim=0)

        # loss coef for the balance of giou and l1
        l1_coef = torch.clip(torch.pow(target_boxes[..., 2:].prod(-1), 0.4), 0.05, 0.95)

        idx = self._get_src_permutation_idx([(torch.unique(i), j) for i, j in indices])
        src_boxes = outputs['pred_boxes'][idx]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['patch_loss_bbox'] = (l1_coef[..., None] * loss_bbox).sum() / num_boxes
        '''loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))'''
        loss_giou = 1 - torch.diag(box_ops.ciou(src_boxes, target_boxes))

        #wh_r = (src_boxes[:, 2:3] + 1e-5) / (src_boxes[:, 3:4] + 1e-5)  # (bsz x n_q) x 1
        #wh_r_cost = (wh_r - 1.45).relu() + (-wh_r + 0.55).relu()

        losses['patch_loss_giou'] = ((1 - l1_coef) * loss_giou).sum() / num_boxes #  + opt.whr_norm_coef*wh_r_cost.sum()

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'patch_labels': self.patch_loss_labels,
            'patch_boxes': self.patch_loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, refine_targets, batc):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             out_patch: dict of tensors
                             -> 'pred_boxes': 'center_x, center_y, w, h'   bsz x n_p_q x 4
             refine_targets: -> targ['boxes']: 'center_x, center_y, w, h'  have been transed to 0 ~ 1 in the last dim,
                                which can be directly computed with sigmoided preds boxes
                             -> 'center_x, center_y' are normalized with patch_w, patch_h
                             -> 'w, h' are normalized with input_w, input_h
                             -> targ['patch_area'] (sigmoided): topx, topy, botx, boty    patchsize x 4
             patch_targets:  -> labels (1 the poor prediction boxes as well as (2 the target boxes fail to locate
                             -> targ['boxes']: 'center_x, center_y, w, h' normalized to 0 ~ 1
                             -> each dict in targets contains 'boxes', 'labels', also with
                               'cls' indicating the error type of the corresponding patch (1 for fp, 0 for fn)
        """
        losses = {}

        # Part I: computing the refine loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        #   Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, refine_targets)
        #   Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(t["boxes"].shape[0] for t in refine_targets if 'boxes' in t)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=0).item()
        #   Compute all the requested losses
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, refine_targets, indices, num_boxes, **kwargs))
        #   In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['pred_gt_mats'] = outputs['pred_gt_mats']
                indices = self.matcher(aux_outputs, refine_targets)

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, refine_targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(refine_targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs):
        """ Perform the computation
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'src_valid_trans': src_valid_trans}
        out_patch = {'pred_logits': patch_cls, 'pred_boxes': patch_box}
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if not opt.test_transpatch:
            out_prob, out_bbox, pre_bbox, valid_trans = \
                outputs['pred_logits'].sigmoid(), outputs['pred_boxes'], outputs['pre_boxes'], outputs['src_valid_trans']

        # patch info extract
        # method 1
        #box = patch_bbox[0][torch.sort(patch_cls[0, :, 1:].max(1).values, dim=0, descending=True).indices[:opt.num_patch_val]]

        """# method 2
        box = patch_bbox[0][patch_cls.max(1).indices[0,torch.sort(patch_cls.max(1).values, dim=1, descending=True).indices[0, :opt.num_patch_val]]]
        box[..., :2] = box[..., :2] - box[..., 2:] / 2
        box[..., 2:] = box[..., :2] + box[..., 2:]
        box[0, :2] = box[:, :2].min(0).values
        box[0, 2:] = box[:, 2:].max(0).values
        patch_bbox = box[0:1]
        
        #patch_bbox = patch_bbox[patch_cls[..., 1] == patch_cls[..., 1].max()]
        patch_area = patch_bbox.detach().clone()
        patch_area = torch.clip(patch_area, min=0, max=1.)
        """

        patch_area = outputs['patch_area']

        # refine info extract
        scores, ind = out_prob[..., 1:2].max(-1)

        batc = {}
        batc['nnpp'] = torch.tensor([-1., -1., 1., 1.], device=patch_area.device)

        # perform nms(boxes: Tensor[N, 4], scores: Tensor[N], iou_threshold: float)
        out_bbox_copy = out_bbox.detach().clone().squeeze(0)
        ct_copy = out_bbox_copy[..., :2]
        bbox_amodal_copy = out_bbox_copy[..., 2:]
        rect_copy = bbox_amodal_copy * batc['nnpp'] + ct_copy.repeat(
            [1 if nu < (len(ct_copy.shape) - 1) else 2 for nu in range(len(ct_copy.shape))])
        keep_ind = nms(boxes=rect_copy, scores=out_prob[0, :, 1], iou_threshold=opt.nms_thre)

        # refer_point_sig, src_valid_trans, bac=None, recti_wh=None, lvls=None
        boxes = pre2samp(out_bbox[:, keep_ind], valid_trans[:, 0:1], bac=batc, lvls=1,
                         recti_wh=torch.tensor([opt.input_w//8, opt.input_h//8], device=patch_area.device))
        pre_boxes = pre2samp(pre_bbox[:, keep_ind], valid_trans[:, 0:1], bac=batc, lvls=1,
                         recti_wh=torch.tensor([opt.input_w//8, opt.input_h//8], device=patch_area.device))
        ct = boxes[..., :2]

        pre_box = pre_boxes[..., 2:] * batc['nnpp'] + pre_boxes[..., :2].repeat(
            [1 if nu < (len(pre_boxes[..., :2].shape) - 1) else 2 for nu in range(len(pre_boxes[..., :2].shape))])

        # ablation on motion
        #tracking = torch.zeros_like(tracking)

        bbox_amodal = boxes[..., 2:]
        rect = bbox_amodal * batc['nnpp'] + ct.repeat([1 if nu<(len(ct.shape)-1) else 2 for nu in range(len(ct.shape))])
        rect = rect.squeeze(-2)
        ct = ct.squeeze(-2)
        #         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        #         scores = topk_values
        #         topk_boxes = topk_indexes // out_logits.shape[2]
        #         labels = topk_indexes % out_logits.shape[2]
        #         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        #         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        results = [{'refine_scores': s, 'refine_boxes': b, 'refine_cts': c, 'refine_tracking': t}
                   for s, b, c, t in zip(scores[:, keep_ind], rect, ct, pre_box)]

        return results, patch_area


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20
    device = torch.device(args.device)
    backbone = build_backbone(args)
    #transpatch = build_transpatch_tnt(args)
    transformer = build_deforamble_transformer(args)
    #defor_detr = torch.load('/home/beeno/pycharm/py_code/CenterTrack/src_model_transfiner_v1/lib/post_transfiner/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth')
    #transformer.load_state_dict(defor_detr['model'], strict=False)
    model = Transfiner(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    patch_matcher = build_patch_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef,
                   'pre_loss_bbox': args.pre_bbox_loss_coef, 'pre_loss_giou': args.pre_giou_loss_coef,
                   'patch_loss_ce': args.p_cls_loss_coef, 'patch_loss_bbox': args.p_bbox_loss_coef, 'patch_loss_giou': args.p_giou_loss_coef}
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if (not 'patch' in k)})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items() if (not 'patch' in k)})
        weight_dict.update(aux_weight_dict)
    if opt.aux_loss_tnt and (not opt.transformer_trainonly):
        aux_weight_dict = {}
        for i in range(opt.depth - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if 'patch' in k})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, patch_matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

def build_eval(args):
    num_classes = 20
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    #defor_detr = torch.load('/home/beeno/pycharm/py_code/CenterTrack/src_model_transfiner_v1/lib/post_transfiner/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth')
    #transformer.load_state_dict(defor_detr['model'], strict=False)
    model = Transfiner(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    patch_matcher = build_patch_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef,
                   'patch_loss_ce': args.p_cls_loss_coef, 'patch_loss_bbox': args.p_bbox_loss_coef, 'patch_loss_giou': args.p_giou_loss_coef}
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if (not 'patch' in k)})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items() if (not 'patch' in k)})
        weight_dict.update(aux_weight_dict)
    if opt.aux_loss_tnt:
        aux_weight_dict = {}
        for i in range(opt.depth - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if 'patch' in k})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, patch_matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


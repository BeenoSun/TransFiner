# Modified by Peize Sun, Rufeng Zhang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math
import torchvision
from post_transfiner.patch_matcher import PatchMatcher
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from post_transfiner.utils.misc import inverse_sigmoid
from post_transfiner.utils.ms_deform_attn import MSDeformAttn
from post_transfiner.utils.utils import hmbased_initialization, pre2samp, generate_refine_gt, boxout2xyxy, eval_pa
from post_transfiner.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from opts import opts
from .Feature_Fetcher import FeatureFetcher
opt = opts().parse()

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 checkpoint_enc_ffn=False, checkpoint_dec_ffn=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.patchmatcher = PatchMatcher()
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          checkpoint_enc_ffn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points,
                                                          checkpoint_dec_ffn)
        fusion_layer = FusionLayer(d_model, dim_feedforward, dropout, activation,
                                   num_feature_levels, nhead, dec_n_points, checkpoint_dec_ffn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                    fusion_layer=fusion_layer)
        #self.decoder_track = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.linear1 = nn.Linear(d_model, d_model*num_feature_levels)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*num_feature_levels, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    # transform memory to a query embed
    def my_forward_ffn(self, query):
        query2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        return self.norm2(query + self.dropout3(query2))

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):

        _, H, W = mask.shape
        valid_sum_h = torch.sum(~mask, 1, keepdim=True)
        valid_H, _ = torch.max(valid_sum_h, dim=2)
        valid_H.squeeze_(1)
        valid_sum_w = torch.sum(~mask, 2, keepdim=True)
        valid_W, _ = torch.max(valid_sum_w, dim=1)
        valid_W.squeeze_(1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio_h = torch.clamp(valid_ratio_h, min=1e-3, max=1.1)
        valid_ratio_w = torch.clamp(valid_ratio_w, min=1e-3, max=1.1)
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)

        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_tgt_embed=None, batc=None,
                pre_srcs=None, pre_masks=None, pre_hms=None, pre_pos_embeds=None):
        assert self.two_stage or query_tgt_embed is not None

        patch2batch = [nu_ for nu_ in range(srcs[0].shape[0])]
        patch_area = torch.tensor([[0., 0., 1., 1.]], device=srcs[0].device).repeat(srcs[0].shape[0], 1) # bsz x 4
        #print('area size:',(patch_area[:, 2:]-patch_area[:,:2]).prod(-1), 'w / h:',(patch_area[:,2]-patch_area[:,0]+1e-5)/(patch_area[:,3]-patch_area[:,1]+1e-5))

        # stretch the bsz dim to the num of patchsz for the convenience of calculation
        # patched_box = patchfilter(patch_box, gt_patch_matched) # also assign whole img area to no tg patch img without affecting patch targets
        batc['nnpp'] = torch.tensor([-1., -1., 1., 1.], device=patch_area.device)
        # generate the gt for refine decoder
        if not opt.eva:
            patch_area = generate_refine_gt(batc, patch_area, patch2batch)
        else:
            batc['pad_mask'] = batc['pad_mask'].to(patch_area.device)
            patch_area = eval_pa(batc, patch_area)

        # input: srcs, masks, pos, query_embeds, patch_cls, patch_box, bch
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []

        # pre
        pre_src_flatten = []
        pre_mask_flatten = []
        pre_lvl_pos_embed_flatten = []

        spatial_shapes = []
        for lvl, (src, mask, pos_embed, pre_src, pre_mask, pre_pos_embed) in enumerate(
                zip(srcs, masks, pos_embeds, pre_srcs, pre_masks, pre_pos_embeds)):
            assert pre_src.shape == src.shape
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

            # pre
            pre_pos_embed = pre_pos_embed.flatten(2).transpose(1, 2)
            pre_lvl_pos_embed_flatten.append(pre_pos_embed + self.level_embed[lvl].view(1, 1, -1))
            pre_src_flatten.append(pre_src.flatten(2).transpose(1, 2))
            pre_mask_flatten.append(pre_mask.flatten(1))

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # pre
        pre_src_flatten = torch.cat(pre_src_flatten, 1)
        pre_mask_flatten = torch.cat(pre_mask_flatten, 1)
        pre_lvl_pos_embed_flatten = torch.cat(pre_lvl_pos_embed_flatten, 1)
        pre_valid_ratios = torch.stack([self.get_valid_ratio(pre_m) for pre_m in pre_masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten)
        with torch.no_grad():
            pre_memory = self.encoder(pre_src_flatten, spatial_shapes, level_start_index, pre_valid_ratios,
                                      pre_lvl_pos_embed_flatten, pre_mask_flatten)

        # match the patch groundtruth generated with the patch prediction (multiple to one matching is allowed)
        #gt_patch_matched, patch2batch, indices_tg = self.patchmatcher(patch_cls, patch_box, batc)

        reference_points, pre_reference_points, mask_flatten, src_valid_trans, query_recon_masks, pred_gt_mats = \
            hmbased_initialization(patch2batch, patch_area, batc, memory, spatial_shapes)

        #query_embed = self.my_forward_ffn(query_embed)

        #memory = memory[patch2batch]

        ps, _, c = memory.shape
        pre_tgt, tgt = torch.split(query_tgt_embed, c, dim=1) # tgt: num_queries x 256
        tgt = tgt[:opt.real_num_queries] # num_queries x 256 -> real_num_queries x 256
        tgt = tgt.unsqueeze(0).expand(ps, -1, -1)
        pre_tgt = pre_tgt[:opt.real_num_queries]
        pre_tgt = pre_tgt.unsqueeze(0).expand(ps, -1, -1)
        init_reference_out = reference_points

        # decoder
        hs, inter_references, pre_hs = \
                            self.decoder(tgt, reference_points, patch_area, memory, spatial_shapes, level_start_index,
                                            valid_ratios[patch2batch], None, mask_flatten, src_valid_trans, batc=batc,
                                            pre_tgt=pre_tgt, pre_reference_points=pre_reference_points,
                                            pre_memory=pre_memory, query_recon_masks=query_recon_masks)
        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out, pre_hs, patch_area, src_valid_trans, query_recon_masks, pred_gt_mats


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 checkpoint_ffn=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # use torch.utils.checkpoint.checkpoint to save memory
        self.checkpoint_ffn = checkpoint_ffn

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        if self.checkpoint_ffn:
            src = torch.utils.checkpoint.checkpoint(self.forward_ffn, src)
        else:
            src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 checkpoint_ffn=False):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffe = FeatureFetcher()
        # use torch.utils.checkpoint.checkpoint to save memory
        self.checkpoint_ffn = checkpoint_ffn

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, pre_refer=None, reconstruct_embed=None):
        # the embed here better to be 'pre' mode for tgt only use sigmoid range from 0~1
        query_ref_boxes_sine_embed = self.ffe(reference=pre_refer, before_cx=True) # bsz x n_q x lvls x 6
        # self attention
        q = k = self.with_pos_embed(tgt, query_ref_boxes_sine_embed+reconstruct_embed)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # semantic aligner
        query_pos = self.ffe(tgt=tgt, memory=src, level_start_index=level_start_index, reference=reference_points)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        if self.checkpoint_ffn:
            tgt = torch.utils.checkpoint.checkpoint(self.forward_ffn, tgt)
        else:
            tgt = self.forward_ffn(tgt)

        return tgt

class FusionLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, checkpoint_ffn=False):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffe = FeatureFetcher()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, pre_tgt, src, pre_src, refer, pre_refer, det_track_pos):
        """
        Args:
            tgt: bsz x n_q x c
            pre_tgt: bsz x n_q x c
            src:
            pre_src:
            refer: bsz x n_q x 6
            pre_refer: bsz x n_q x 6
            det_track_pos: 1 x (n_q x 2) x c
        Returns:

        """
        # the embed here better to be 'pre' mode for tgt only use sigmoid range from 0~1
        det_track_tgt = torch.cat([tgt, pre_tgt], dim=1) # bsz x (n_qx2) x 256
        sine_embed = self.ffe(reference=refer, before_cx=True) # bsz x n_q x 256
        sine_embed_pre = self.ffe(reference=pre_refer, before_cx=True) # bsz x n_q x 256
        det_track_emb = torch.cat([sine_embed, sine_embed_pre], dim=1) # bsz x (n_qx2) x 256
        det_track_emb = det_track_emb + det_track_pos

        # atten mask for fusion of det & track
        att_mask = torch.tensor([[-1e8, -10],
                                 [-10, -1e8]], device=det_track_emb.device) # default: -10
        att_mask = F.interpolate(att_mask[None, None].float(),
                                 size=[det_track_tgt.shape[1], det_track_tgt.shape[1]]).squeeze()
        att_mask_mask = (1 - torch.cat([
            torch.cat([torch.zeros(tgt.shape[1], tgt.shape[1]), torch.eye(tgt.shape[1])], dim=1),
            torch.cat([torch.eye(tgt.shape[1]), torch.zeros(tgt.shape[1], tgt.shape[1])], dim=1)
        ], dim=0)).to(att_mask.device)
        att_mask = (att_mask * att_mask_mask).float()

        # self fusion attention
        q = k = self.with_pos_embed(det_track_tgt, det_track_emb) # bsz x (n_qx2) x 256
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), det_track_tgt.transpose(0, 1),
                              attn_mask = att_mask)[0].transpose(0, 1)
        det_track_tgt = det_track_tgt + self.dropout2(tgt2)
        det_track_tgt = self.norm2(det_track_tgt)
        tgt, pre_tgt = torch.split(det_track_tgt, [tgt.shape[1], tgt.shape[1]], dim=1)

        return tgt, pre_tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, fusion_layer=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.layers_pre = _get_clones(decoder_layer, num_layers)
        self.fusion_layers = _get_clones(fusion_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.motion_embed = None
        self.class_embed = None
        #self.det_track_pos = nn.Parameter(torch.Tensor(opt.real_num_queries * 2, 256))
        self.reconstruct_embed = nn.Parameter(torch.Tensor(1, 256))
        self.matching_embed = nn.Parameter(torch.Tensor(1, 256))

    def forward(self, tgt, reference_points, patch_area, src, src_spatial_shapes, level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, src_valid_trans=None, batc=None, pre_tgt=None,
                pre_reference_points=None, pre_memory=None, query_recon_masks=None):
        output = tgt
        pre_out = pre_tgt

        intermediate, intermediate_reference_points = [], []
        pre_intermediate, pre_intermediate_reference_points = [], []
        for lid, (layer, layer_pre, fusion_layer) in enumerate(zip(self.layers, self.layers_pre, self.fusion_layers)):
            if reference_points.shape[-1] == (4 if not opt.box_amodal_pred else 6):
                # -> psz x n_q x lvls x 6
                reference_points_input = pre2samp(reference_points, src_valid_trans, batc, lvls=4,
                                    recti_wh=torch.tensor([opt.input_w//8, opt.input_h//8], device=patch_area.device))
                pre_reference_points_input = pre2samp(pre_reference_points, src_valid_trans, batc, lvls=4,
                                    recti_wh=torch.tensor([opt.input_w//8, opt.input_h//8], device=patch_area.device))

                #reference_points_input = reference_points[:, :, None] \
                 #                        * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = pre2samp(reference_points, src_valid_trans, batc, lvls=4,
                                    recti_wh=torch.tensor([opt.input_w//8, opt.input_h//8], device=patch_area.device))
                pre_reference_points_input = pre2samp(pre_reference_points, src_valid_trans, batc, lvls=4,
                                    recti_wh=torch.tensor([opt.input_w//8, opt.input_h//8], device=patch_area.device))
                #reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            recon_match_embed = query_recon_masks[..., None].float() * self.reconstruct_embed[:, None] + \
                                (1-query_recon_masks[..., None].float()) * self.matching_embed[:, None] # bsz x n_q x c

            output, pre_out = fusion_layer(output, pre_out, src, pre_memory, reference_points.detach().clone(),
                                           pre_reference_points.detach().clone(),
                                           recon_match_embed.repeat(1, 2, 1))

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                           level_start_index, src_padding_mask, pre_refer=reference_points.detach().clone(),
                           reconstruct_embed=recon_match_embed)

            pre_out = layer_pre(pre_out, query_pos, pre_reference_points_input, pre_memory, src_spatial_shapes,
                                level_start_index, src_padding_mask, pre_refer=pre_reference_points.detach().clone(),
                                reconstruct_embed=recon_match_embed)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                pre_tmp = self.motion_embed[lid](pre_out)
                if reference_points.shape[-1] == (4 if not opt.box_amodal_pred else 6):
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()

                    new_reference_points_pre = pre_tmp + inverse_sigmoid(reference_points)
                    new_reference_points_pre = new_reference_points_pre.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                pre_reference_points = new_reference_points_pre.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

                # pre
                pre_intermediate.append(pre_out)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(pre_intermediate)
        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        checkpoint_enc_ffn=args.checkpoint_enc_ffn,
        checkpoint_dec_ffn=args.checkpoint_dec_ffn
    )
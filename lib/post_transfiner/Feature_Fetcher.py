import torch
import torchvision
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from opts import opts
from .utils.utils import boxout2xyxy
from .utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import math
opt = opts().parse()

class FeatureFetcher(nn.Module):

    def __init__(self, lvls=opt.num_feature_levels):

        super().__init__()
        self.lvls = lvls
        self.lvl_hws = [[(torch.tensor(opt.input_h / 8) / 2 ** sca).ceil().int(),
                         (torch.tensor(opt.input_w / 8) / 2 ** sca).ceil().int()] for sca in range(lvls)]
        self.point1 = nn.Sequential(
            nn.Conv2d(opt.hidden_dim, opt.hidden_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.point2 = nn.Sequential(
            nn.Linear(opt.hidden_dim // 4 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, opt.nheads * 2),
        )
        nn.init.constant_(self.point2[-1].weight.data, 0)
        nn.init.constant_(self.point2[-1].bias.data, 0)
        self.attn1 = nn.Linear(opt.hidden_dim, opt.hidden_dim * opt.nheads)
        self.attn2 = nn.Linear(opt.hidden_dim, 1)

    def forward(self, tgt=None, memory=None, level_start_index=None, reference=None, before_cx=False):
        """

        Args:
            tgt: bsz x n_q x c
            memory: bsz x total_lvl x c
            level_start_index:
            reference: bsz x n_q x lvls x 6

        Returns:
            query_embed: bsz x n_q x c
        """
        if before_cx:
            #reference_boxes_xyxy = boxout2xyxy(reference.detach().clone()[:, :, 0]) # bsz x n_q x lvls x 6 -> bsz x n_q x 4
            #reference_boxes_cxcywh = box_xyxy_to_cxcywh(reference_boxes_xyxy)
            query_ref_boxes_sine_embed = self.gen_sineembed_for_position(reference[..., :2]) # bsz x n_q x 256
            return query_ref_boxes_sine_embed
        else:
            query_embed = []
            for lvi in range(self.lvls):
                bs, n_model, num_queries = memory.shape[0], memory.shape[-1], torch.tensor(opt.real_num_queries, device=memory.device)
                mem = memory[:, level_start_index[lvi]:level_start_index[lvi+1], :] if lvi!=(self.lvls-1) else memory[:, level_start_index[lvi]:, :] # bsz x lvl x c
                memory_h, memory_w = self.lvl_hws[lvi][0].to(mem.device), self.lvl_hws[lvi][1].to(mem.device)
                mem_2d = mem.transpose(1, 2).view(bs, memory.shape[-1], memory_h, memory_w) # bsz x c x lvl_h x lvl_w

                refer_sam = reference[:, :, lvi] # bsz x n_q x 6
                # bsz x n_q x 6 (sample) -> bsz x n_q x 4 (sample)
                reference_boxes_xyxy = boxout2xyxy(refer_sam)
                reference_boxes_xyxy = torch.clip(reference_boxes_xyxy, min=0., max=1.)
                reference_boxes_xyxy[:, :, [0, 2]] *= memory_w
                reference_boxes_xyxy[:, :, [1, 3]] *= memory_h

                # extract the ROI through roialign
                q_content = torchvision.ops.roi_align(mem_2d,
                    list(torch.unbind(reference_boxes_xyxy, dim=0)),
                    output_size=(7, 7), spatial_scale=1.0, aligned=True) # (bs * num_queries, c, 7, 7)

                q_content_points = torchvision.ops.roi_align(mem_2d,
                    list(torch.unbind(reference_boxes_xyxy, dim=0)),
                    output_size=(7, 7), spatial_scale=1.0, aligned=True) # (bs * num_queries, c, 7, 7)

                q_content_index = q_content_points.view(bs * num_queries, -1, 7, 7) # (bsz x n_q) x c x 7 x 7

                points = self.point1(q_content_index) # (bsz x n_q) x 256 x 7 x 7 -> (bsz x n_q) x 64 x 7 x 7
                points = points.reshape(bs * num_queries, -1) # (bsz x n_q) x (64 x 7 x 7)
                points = self.point2(points) # (bsz x n_q) x (64 x 7 x 7) -> (bsz x n_q) x (nheads x 2)
                points = points.view(bs * num_queries, 1, opt.nheads, 2).tanh() # (bsz * n_q) x 1 x nhead x 2

                q_content = F.grid_sample(q_content, points, padding_mode="zeros", align_corners=False).view(
                    bs * num_queries, -1)  # points: offset to sample features
                q_content = q_content.view(bs, num_queries, -1, 8).transpose(3, 2) # (bsz, n_q, n_head, 256)
                q_content = q_content * self.attn1(tgt).view(bs, num_queries, opt.nheads, n_model).sigmoid() # (bsz, n_q, n_head, 256)
                query_embed.append(q_content)

            query_embed = torch.stack(query_embed, dim=2).flatten(2, 3) # (bsz, n_q, lvls*n_head, 256)
            lvlhead_w = F.softmax(self.attn2(query_embed).squeeze(-1), -1)[..., None]

            return (query_embed * lvlhead_w).sum(-2) # (bsz, n_q, 256)


    @torch.no_grad()
    def gen_sineembed_for_position(self, pos_tensor):
        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos

"""
Detector model and criterion classes.
"""
from post_transfiner.utils import box_ops
from post_transfiner.utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import copy
from torch.cuda.amp import autocast, GradScaler
from opts import opts
opt = opts().parse()
from .backbone import *



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

class Detector(nn.Module):
    def __init__(self, num_classes, pre_trained=None, det_token_num=100, backbone_name='tiny', init_pe_size=[800,1344], mid_pe_size=None, use_checkpoint=False):
        super().__init__()
        # import pdb;pdb.set_trace()
        if backbone_name == 'tiny':
            self.backbone, hidden_dim = tiny(pretrained=pre_trained)
        elif backbone_name == 'small':
            self.backbone, hidden_dim = small(pretrained=pre_trained)
        elif backbone_name == 'base':
            self.backbone, hidden_dim = base(pretrained=pre_trained)
        elif backbone_name == 'small_dWr':
            self.backbone, hidden_dim = small_dWr(pretrained=pre_trained)
        else:
            raise ValueError(f'backbone {backbone_name} not supported')
        
        self.backbone.finetune_det(det_token_num=det_token_num, img_size=init_pe_size, mid_pe_size=mid_pe_size, use_checkpoint=use_checkpoint)
        
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.class_embed = self.class_embed if not opt.aux_loss_tnt else nn.ModuleList([copy.deepcopy(self.class_embed) for _ in range(opt.depth)])
        self.bbox_embed = self.bbox_embed if not opt.aux_loss_tnt else nn.ModuleList([copy.deepcopy(self.bbox_embed) for _ in range(opt.depth)])

    @autocast(enabled=opt.withamp)
    def forward(self, samples: NestedTensor):
        # import pdb;pdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        x = self.backbone(samples.tensors)
        # x = x[:, 1:,:]
        outputs_class = self.class_embed(x) if not opt.aux_loss_tnt else \
            torch.stack([self.class_embed[i](x[i]) for i in range(opt.depth)], dim=0)
        outputs_coord = self.bbox_embed(x).sigmoid() if not opt.aux_loss_tnt else \
            torch.stack([self.bbox_embed[i](x[i]).sigmoid() for i in range(opt.depth)], dim=0)

        return outputs_class, outputs_coord

    def forward_return_attention(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        attention = self.backbone(samples.tensors, return_attention=True)
        return attention


def build_patchdet(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20
    device = torch.device(args.device)

    # import pdb;pdb.set_trace()
    model = Detector(
        num_classes=num_classes,
        pre_trained=args.pre_trained,
        det_token_num=args.det_token_num,
        backbone_name=args.backbone_name,
        init_pe_size=args.init_pe_size,
        mid_pe_size=args.mid_pe_size,
        use_checkpoint=args.use_checkpoint,

    )


    return model


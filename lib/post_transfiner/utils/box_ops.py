# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import math

def box_cxcyhw_to_xyxy(x):
    x_c, y_c, h, w = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    union[union == 0] = 1e-12
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    area[area == 0] = 1e-12

    return iou - (area - union) / area

def ciou(bboxes1, bboxes2):
    # n x 4 / m x 4  (sigmoided cxcywh)
    #bboxes1 = torch.sigmoid(bboxes1)
    #bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0] # n
    cols = bboxes2.shape[0] # m
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2]) # n
    h1 = torch.exp(bboxes1[:, 3]) # n
    w2 = torch.exp(bboxes2[:, 2]) # m
    h2 = torch.exp(bboxes2[:, 3]) # m
    area1 = w1 * h1 # n
    area2 = w2 * h2 # m
    center_x1 = bboxes1[:, 0] # n
    center_y1 = bboxes1[:, 1] # n
    center_x2 = bboxes2[:, 0] # m
    center_y2 = bboxes2[:, 1] # m

    inter_l = torch.max((center_x1 - w1 / 2)[None, :, None], (center_x2 - w2 / 2)[:, None, None]) # m x n x 1
    inter_r = torch.min((center_x1 + w1 / 2)[None, :, None], (center_x2 + w2 / 2)[:, None, None])
    inter_t = torch.max((center_y1 - h1 / 2)[None, :, None], (center_y2 - h2 / 2)[:, None, None])
    inter_b = torch.min((center_y1 + h1 / 2)[None, :, None], (center_y2 + h2 / 2)[:, None, None])
    inter_area = (torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)).squeeze(-1) # m x n

    c_l = torch.min((center_x1 - w1 / 2)[None, :, None], (center_x2 - w2 / 2)[:, None, None]) # m x n x 1
    c_r = torch.max((center_x1 + w1 / 2)[None, :, None], (center_x2 + w2 / 2)[:, None, None])
    c_t = torch.min((center_y1 - h1 / 2)[None, :, None], (center_y2 - h2 / 2)[:, None, None])
    c_b = torch.max((center_y1 + h1 / 2)[None, :, None], (center_y2 + h2 / 2)[:, None, None])

    inter_diag = ((center_x2[:, None] - center_x1[None])**2 + (center_y2[:, None] - center_y1[None])**2) # m x n
    c_diag = (torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2).squeeze(-1) # m x n

    union = area1[None] + area2[:, None] - inter_area # m x n
    u = (inter_diag) / (c_diag + torch.tensor(1e-5, device=inter_diag.device))
    iou = (inter_area) / (union + torch.tensor(1e-5, device=inter_diag.device))
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / (h2 + torch.tensor(1e-5, device=inter_diag.device)))[:, None] -
                                          torch.atan(w1 / ((h1 + torch.tensor(1e-5, device=inter_diag.device))))[None]), 2) # m x n
    with torch.no_grad():
        S = (iou>0.5).float() # m x n
        alpha= S*v/(1-iou+v+torch.tensor(1e-5, device=inter_diag.device)) # m x n
    cious = iou - u - alpha * v
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if not exchange:
        cious = cious.T

    return cious

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

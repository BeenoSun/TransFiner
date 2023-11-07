from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict
import seaborn_image as isns
import pycocotools.coco as coco
import torch
import torch.utils.data as data
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import copy

class GenericDataset(data.Dataset):
  is_fusion_dataset = False
  default_resolution = None
  num_categories = None
  class_name = None
  # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
  # Not using 0 because 0 is used for don't care region and ignore loss.
  cat_ids = None
  max_objs = None
  rest_focal_length = 1200
  num_joints = 17
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
           [4, 6], [3, 5], [5, 6], 
           [5, 7], [7, 9], [6, 8], [8, 10], 
           [6, 12], [5, 11], [11, 12], 
           [12, 14], [14, 16], [11, 13], [13, 15]]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                      dtype=np.float32)
  _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
  ignore_val = 1
  nuscenes_att_range = {0: [0, 1], 1: [0, 1], 2: [2, 3, 4], 3: [2, 3, 4], 
    4: [2, 3, 4], 5: [5, 6, 7], 6: [5, 6, 7], 7: [5, 6, 7]}
  def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
    super(GenericDataset, self).__init__()
    if opt is not None and split is not None:
      self.split = split
      self.opt = opt
      self._data_rng = np.random.RandomState(123)
    
    if ann_path is not None and img_dir is not None:
      print('==> initializing {} data from {}, \n images from {} ...'.format(
        split, ann_path, img_dir))
      self.coco = coco.COCO(ann_path)
      self.images = self.coco.getImgIds()

      if opt.tracking:
        if not ('videos' in self.coco.dataset):
          self.fake_video_data()
        print('Creating video index!')
        self.video_to_images = defaultdict(list)
        for image in self.coco.dataset['images']:
          self.video_to_images[image['video_id']].append(image)
      
      self.img_dir = img_dir

  def __getitem__(self, index):
    opt = self.opt
    img, anns, img_info, img_path = self._load_data(index)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)
    aug_s, rot, flipped = 1, 0, 0
    if self.split == 'train':
      c, aug_s, rot = self._get_aug_param(c, s, width, height) # not affected by static training here
      s = s * aug_s
      if np.random.random() < opt.flip:
        flipped = 1
        img = img[:, ::-1, :]
        anns = self._flip_anns(anns, width)
    # Geometric contraction, expansion, dilation, reflection, rotation, shear,
    # similarity transformations, spiral similarities, and translation are all
    # affine transformations
    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w//2, opt.output_h//2])

    # addition for higres
    ori_trans_input = get_affine_transform(c, s, 0, [opt.ori_input_w, opt.ori_input_h])
    ori_inp_image = cv2.warpAffine(img, ori_trans_input, (opt.ori_input_w, opt.ori_input_h),
                                   flags=cv2.INTER_LINEAR)
    #ori_inp_image = ((ori_inp_image / 255. - self.mean) / self.std).astype(np.float32)
    #ori_images = ori_inp_image.transpose(2, 0, 1).reshape(1, 3, opt.ori_input_h, opt.ori_input_w)

    inp, padding_mask, inp_ori, ori_inp_image = \
      self._get_input(img, trans_input, ori=True, ori_inp_image=ori_inp_image) # 0 for object, 1 for objectless

    ret = {'image': inp, 'pad_mask': padding_mask.astype(np.bool), 'ori_inp_image': ori_inp_image}
    ret['iscrowdhuman'] = ('crowdhuman_val' in img_path) or ('CrowdHuman_train' in img_path)
    ret['inp_ori'] = inp_ori
    #ret = {'cur_img': img}
    #hm_withnoise = self._get_pre_dets(
     # anns, trans_input, trans_output, hm_plot=True, other_clues=False)
    #ret['hm_withnoise'] = hm_withnoise
    ret['cur_meta'] = {'c': copy.deepcopy(c), 's': copy.deepcopy(s), 'height': height, 'width': width,
                 'out_height': opt.output_h, 'out_width': opt.output_w,
                 'inp_height': opt.input_h, 'inp_width': opt.input_w,
                 'trans_input': copy.deepcopy(trans_input), 'trans_output': copy.deepcopy(trans_output)}
    anns_varlen = torch.stack([torch.tensor(ann['bbox']).float() for ann in anns])
    pad_len = self.max_objs - anns_varlen.shape[0] if (self.max_objs - anns_varlen.shape[0]) > 0 else 0
    ret['anns'] = torch.cat([anns_varlen[:(self.max_objs-pad_len)], anns_varlen.new_zeros((pad_len, 4))])
#    ret['img_cat'] = int(img_info['file_name'][6:8])
    gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

    pre_cts, track_ids = None, None
    if opt.tracking:
      # pre one
      pre_image, pre_anns, pre_img_info, frame_dist = self._load_pre_data(
        img_info['video_id'], img_info['frame_id'],
        img_info['sensor_id'] if 'sensor_id' in img_info else 1)
      if flipped:
        pre_image = pre_image[:, ::-1, :].copy()
        pre_anns = self._flip_anns(pre_anns, width)
      assert (img_info['file_name'] == pre_img_info['file_name']) or (not ret['iscrowdhuman'])
      if opt.same_aug_pre and frame_dist != 0:
        trans_input_pre = trans_input
        trans_output_pre = trans_output
        c_pre, s_pre = c, s
      else:
        c_pre, aug_s_pre, _ = self._get_aug_param(
          c, s, width, height, disturb=True)
        s_pre = s * aug_s_pre
        trans_input_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.input_w, opt.input_h])
        trans_output_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.output_w//2, opt.output_h//2])

      # addition for higres
      ori_trans_input_pre = get_affine_transform(c_pre, s_pre, 0, [opt.ori_input_w, opt.ori_input_h])
      ori_inp_image_pre = cv2.warpAffine(pre_image, ori_trans_input_pre, (opt.ori_input_w, opt.ori_input_h),
                                     flags=cv2.INTER_LINEAR)

      pre_img, pre_padding_mask, ori_inp_image_pre = self._get_input(pre_image, trans_input_pre, ori_inp_image=ori_inp_image_pre)
      _, pre_cts, pre_bxs, track_ids = self._get_pre_dets(
        pre_anns, trans_input_pre, trans_output_pre, hm_plot=True, other_clues=True, down=True)
      pre_hm = self._get_pre_dets(
        pre_anns, ori_trans_input_pre, trans_output_pre, hm_plot=True, other_clues=False, ori=True)
      ret['ori_inp_image_pre'] = ori_inp_image_pre
      ret['pre_img'] = pre_img
      ret['pre_pad_mask'] = pre_padding_mask.astype(np.bool)
      ret['pre_hm'] = pre_hm
      #ret['pre_meta'] = {'c': copy.deepcopy(c_pre), 's': copy.deepcopy(s_pre), 'height': height, 'width': width,
       #                  'out_height': opt.output_h, 'out_width': opt.output_w,
        #                 'inp_height': opt.input_h, 'inp_width': opt.input_w,
         #                'trans_input': copy.deepcopy(trans_input_pre), 'trans_output': copy.deepcopy(trans_output_pre)}

      # one before pre
      """pre2_image, pre2_anns, _, frame_dist = self._load_pre_data(
        pre_img_info['video_id'], pre_img_info['frame_id'],
        pre_img_info['sensor_id'] if 'sensor_id' in pre_img_info else 1)
      if flipped:
        pre2_image = pre2_image[:, ::-1, :].copy()
        pre2_anns = self._flip_anns(pre2_anns, width)
      if opt.same_aug_pre and frame_dist != 0:
        trans_input_pre2 = trans_input_pre
        trans_output_pre2 = trans_output_pre
      else:
        c_pre2, aug_s_pre2, _ = self._get_aug_param(
          c, s, width, height, disturb=True)
        s_pre2 = s * aug_s_pre2
        trans_input_pre2 = get_affine_transform(
          c_pre2, s_pre2, rot, [opt.input_w, opt.input_h])
        trans_output_pre2 = get_affine_transform(
          c_pre2, s_pre2, rot, [opt.output_w, opt.output_h])
      pre2_img, _ = self._get_input(pre2_image, trans_input_pre2)
      pre2_hm = self._get_pre_dets(
        pre2_anns, trans_input_pre2, trans_output_pre2, hm_plot=True, other_clues=False)
      ret['pre2_img'] = pre2_img
      if opt.pre_hm:
        ret['pre2_hm'] = pre2_hm"""

    ### init samples
    self._init_ret(ret, gt_det)
    calib = self._get_calib(img_info, width, height)

    num_objs = min(len(anns), self.max_objs)
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -999:
        ret['anns'][k] = torch.zeros(4, device=ret['anns'].device)
        continue
      bbox, bbox_amodal = self._get_bbox_output(
        ann['bbox'], trans_output, height, width)
      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        self._mask_ignore_or_crowd(ret, cls_id, bbox)
        ret['anns'][k] = torch.zeros(4, device=ret['anns'].device)
        continue
      self._add_instance(
        ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s,
        calib, pre_cts, pre_bxs, track_ids)

    if self.opt.debug > 0:
      gt_det = self._format_gt_det(gt_det)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
              'img_path': img_path, 'calib': calib,
              'flipped': flipped}
      ret['meta'] = meta

    return ret

  def get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib

  def _load_image_anns(self, img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  def _load_data(self, index):
    coco = self.coco
    img_dir = self.img_dir
    img_id = self.images[index]
    
    img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

    return img, anns, img_info, img_path


  def _load_pre_data(self, video_id, frame_id, sensor_id=1):
    img_infos = self.video_to_images[video_id]
    # If training, random sample nearby frames as the "previoud" frame
    # If testing, get the exact prevous frame
    if 'train' in self.split:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
          if abs(img_info['frame_id'] - frame_id) < self.opt.max_frame_dist \
             and (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)] # modi for ch # and abs(img_info['frame_id'] - frame_id) != 0
    else:
      img_ids = [(img_info['id'], img_info['frame_id']) \
          for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == -1 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
      if len(img_ids) == 0:
        img_ids = [(img_info['id'], img_info['frame_id']) \
            for img_info in img_infos \
            if (img_info['frame_id'] - frame_id) == 0 and \
            (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
    rand_id = np.random.choice(len(img_ids))
    img_id, pre_frame_id = img_ids[rand_id]
    frame_dist = abs(frame_id - pre_frame_id)
    img, anns, img_info, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
    return img, anns, img_info, frame_dist


  def _get_pre_dets(self, anns, trans_input, trans_output, hm_plot, other_clues, down=False, ori=False):
    down_ratio = self.opt.down_ratio
    return_hm = self.opt.pre_hm
    if down:
      hm_h_out, hm_w_out = int(self.opt.input_h//8), int(self.opt.input_w//8)
      pre_hm_down = np.zeros((1, hm_h_out, hm_w_out), dtype=np.float32) if hm_plot else None
      wh_map = np.zeros((1, hm_h_out, hm_w_out), dtype=np.float32) if hm_plot else None
    if ori:
      hm_h, hm_w = int(self.opt.ori_input_h), int(self.opt.ori_input_w)
    else:
      hm_h, hm_w = int(self.opt.input_h), int(self.opt.input_w)
    trans = trans_input
    pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if hm_plot else None
    pre_cts, track_ids = [], []
    pre_bxs = []
    for ann in anns:
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -99 or \
              ('iscrowd' in ann and ann['iscrowd'] > 0):
        continue
      bbox = self._coco_box_to_bbox(ann['bbox'])  # xywh => x1y1x2y2
      bbox_out = copy.deepcopy(bbox)
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if down:
        bbox_out[:2] = affine_transform(bbox_out[:2], trans_output)
        bbox_out[2:] = affine_transform(bbox_out[2:], trans_output)
        bbox_out_amodal = copy.deepcopy(bbox_out)
        bbox_out[[0, 2]] = np.clip(bbox_out[[0, 2]], 0, hm_w_out - 1)
        bbox_out[[1, 3]] = np.clip(bbox_out[[1, 3]], 0, hm_h_out - 1)
        h_out, w_out = bbox_out[3] - bbox_out[1], bbox_out[2] - bbox_out[0]
      max_rad = 1
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        if down:
          radius_out = gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
          radius_out = max(0, int(radius_out))
          ct_out = np.array(
            [(bbox_out[0] + bbox_out[2]) / 2, (bbox_out[1] + bbox_out[3]) / 2], dtype=np.float32)
          ct0_out = ct_out.copy()
        max_rad = max(max_rad, radius)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct0 = ct.copy()
        conf = 1

        ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
        ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
        conf = 1 if np.random.random() > self.opt.lost_disturb else 0
        if down:
          ct_out = copy.deepcopy(ct) / down_ratio / 2
          ct_out_int = ct_out.astype(np.int32)

        ct_int = ct.astype(np.int32)

        pre_cts.append(ct0 / down_ratio / 2)
        if down:
          pre_bxs.append(bbox_out_amodal)

        track_ids.append(ann['track_id'] if 'track_id' in ann else -1)
        if hm_plot:
          draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)
          if down:
            draw_umich_gaussian(pre_hm_down[0], ct_out_int, radius_out, k=conf)
            wh_map[0, ct_out_int[1], ct_out_int[0]] = radius_out

        if np.random.random() < self.opt.fp_disturb and return_hm:
          ct2 = ct0.copy()
          # Hard code heatmap disturb ratio, haven't tried other numbers.
          ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
          ct2[1] = ct2[1] + np.random.randn() * 0.05 * h 
          ct2_int = ct2.astype(np.int32)
          draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)
          '''if down:
            ct2_out = copy.deepcopy(ct2) / down_ratio
            ct2_out_int = ct2_out.astype(np.int32)
            draw_umich_gaussian(pre_hm_down[0], ct2_out_int, radius_out, k=conf)
            wh_map[0, ct2_out_int[1], ct2_out_int[0]] = radius_out'''

    if hm_plot and (not other_clues):
      if down:
        return pre_hm, pre_hm_down, wh_map
      else:
        return pre_hm
    elif (not hm_plot) and other_clues:
      return pre_cts, track_ids
    else:
      return pre_hm, pre_cts, pre_bxs, track_ids

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


  def _get_aug_param(self, c, s, width, height, disturb=False):
    if (not self.opt.not_rand_crop) and not disturb:
      aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
      w_border = self._get_border(128, width) # 576 for mot    modi for ch
      h_border = self._get_border(128, height) # 576 for mot     modi for ch
      c[0] = np.random.randint(low=w_border, high=width - w_border)
      c[1] = np.random.randint(low=h_border, high=height - h_border)
    else:
      sf = self.opt.scale
      cf = self.opt.shift
      #if type(s) == float:
       # s = [s, s]
      c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      aug_s = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    
    if np.random.random() < self.opt.aug_rot:
      rf = self.opt.rotate
      rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
    else:
      rot = 0
    
    return c, aug_s, rot


  def _flip_anns(self, anns, width):
    for k in range(len(anns)):
      bbox = anns[k]['bbox']
      anns[k]['bbox'] = [
        width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]
      
      if 'hps' in self.opt.heads and 'keypoints' in anns[k]:
        keypoints = np.array(anns[k]['keypoints'], dtype=np.float32).reshape(
          self.num_joints, 3)
        keypoints[:, 0] = width - keypoints[:, 0] - 1
        for e in self.flip_idx:
          keypoints[e[0]], keypoints[e[1]] = \
            keypoints[e[1]].copy(), keypoints[e[0]].copy()
        anns[k]['keypoints'] = keypoints.reshape(-1).tolist()

      if 'rot' in self.opt.heads and 'alpha' in anns[k]:
        anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                           else - np.pi - anns[k]['alpha']

      if 'amodel_offset' in self.opt.heads and 'amodel_center' in anns[k]:
        anns[k]['amodel_center'][0] = width - anns[k]['amodel_center'][0] - 1

      if self.opt.velocity and 'velocity' in anns[k]:
        anns[k]['velocity'] = [-10000, -10000, -10000]

    return anns


  def _get_input(self, img, trans_input, padding_mask=None, ori=False, ori_inp_image=None):
    '''ori_inp_image = cv2.warpAffine(img, ori_trans_input, (opt.ori_input_w, opt.ori_input_h),
                                   flags=cv2.INTER_LINEAR)'''
    img = img.copy()
    if padding_mask is None:
      padding_mask = np.ones_like(img)
    inp = cv2.warpAffine(img, trans_input,
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    if ori:
      inp_ori = copy.deepcopy(inp)
    # to mask = 1 (padding part), not to mask = 0
    affine_padding_mask = cv2.warpAffine(padding_mask, trans_input,
                                         (self.opt.input_w, self.opt.input_h),
                                         flags=cv2.INTER_LINEAR)
    affine_padding_mask = affine_padding_mask[:, :, 0]
    affine_padding_mask[affine_padding_mask > 0] = 1

    inp = (inp.astype(np.float32) / 255.)
    ori_inp_image = (ori_inp_image.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug and np.random.rand() < 0.2:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      color_aug(self._data_rng, ori_inp_image, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    ori_inp_image = (ori_inp_image - self.mean) / self.std
    ori_inp_image = ori_inp_image.transpose(2, 0, 1)
    if ori:
      return inp, 1 - affine_padding_mask, inp_ori, ori_inp_image#.astype(np.float32) # 0 for object, 1 for objectless
    else:
      return inp, 1 - affine_padding_mask, ori_inp_image

  def _init_ret(self, ret, gt_det):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)
    ret['hm_out_hw'] = np.zeros(
      (2, self.opt.output_h, self.opt.output_w),
      np.float32)
    ret['hm_out_amodal'] = np.zeros(
      (4, self.opt.output_h, self.opt.output_w),
      np.float32)
    ret['pre_bxs'] = np.zeros(
      (max_objs, 6), dtype=np.float32)
    #ret['hm_regu_radius'] = np.zeros(
     # (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
      #np.float32)
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)

    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
      'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2, 
      'dep': 1, 'dim': 3, 'amodel_offset': 2}

    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []

    if 'hm_hp' in self.opt.heads:
      num_joints = self.num_joints
      ret['hm_hp'] = np.zeros(
        (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
      ret['hm_hp_mask'] = np.zeros(
        (max_objs * num_joints), dtype=np.float32)
      ret['hp_offset'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
      ret['hp_offset_mask'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)
    
    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
      gt_det.update({'rot': []})


  def _get_calib(self, img_info, width, height):
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _ignore_region(self, region, ignore_val=1.0001):
    np.maximum(region, ignore_val, out=region)


  def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
    # mask out crowd region, only rectangular mask is supported
    if cls_id == 0: # ignore all classes
      self._ignore_region(ret['hm'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                        int(bbox[0]): int(bbox[2]) + 1])
    else:
      # mask out one specific class
      self._ignore_region(ret['hm'][abs(cls_id) - 1, 
                                    int(bbox[1]): int(bbox[3]) + 1, 
                                    int(bbox[0]): int(bbox[2]) + 1])
    if ('hm_hp' in ret) and cls_id <= 1:
      self._ignore_region(ret['hm_hp'][:, int(bbox[1]): int(bbox[3]) + 1, 
                                          int(bbox[0]): int(bbox[2]) + 1])


  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox


  def _get_bbox_output(self, bbox, trans_output, height, width):
    bbox = self._coco_box_to_bbox(bbox).copy() # topx, topy, botx, boty

    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    for t in range(4):
      rect[t] =  affine_transform(rect[t], trans_output)
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

    bbox_amodal = copy.deepcopy(bbox)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w//2 - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h//2 - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    return bbox, bbox_amodal

  def _add_instance(
    self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
    aug_s, calib, pre_cts=None, pre_bxs=None, track_ids=None):
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    if h <= 0 or w <= 0: # filter out the object which is totally out of img
      return
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))
    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)

    ret['hm_out_hw'][:, ct_int[1], ct_int[0]] = [math.ceil(h), math.ceil(w)]

    ret['cat'][k] = cls_id - 1
    ret['mask'][k] = 1
    if 'wh' in ret:
      ret['wh'][k] = 1. * w, 1. * h
      ret['wh_mask'][k] = 1
    # ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)
    #if round(np.sqrt(radius)) < 4:
     # radius_regu = radius
    #else:
     # radius_regu = 4
    #draw_umich_gaussian(ret['hm_regu_radius'][cls_id - 1], ct_int, round(np.sqrt(radius)))
    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    gt_det['scores'].append(1)
    gt_det['clses'].append(cls_id - 1)
    gt_det['cts'].append(ct)

    if 'tracking' in self.opt.heads:
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        pre_bx = pre_bxs[track_ids.index(ann['track_id'])]
        ret['tracking_mask'][k] = 1
        ret['tracking'][k] = pre_ct - ct
        ret['pre_bxs'][k] = pre_ct[0], pre_ct[1], pre_ct[0]-pre_bx[0], pre_ct[1]-pre_bx[1], \
                            pre_bx[2] - pre_ct[0], pre_bx[3] - pre_ct[1]
        gt_det['tracking'].append(ret['tracking'][k])
      else:
        ret['pre_bxs'][k] = ct[0], ct[1], ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
        gt_det['tracking'].append(np.zeros(2, np.float32))

    if 'ltrb' in self.opt.heads:
      ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
        bbox[2] - ct_int[0], bbox[3] - ct_int[1]
      ret['ltrb_mask'][k] = 1

    if 'ltrb_amodal' in self.opt.heads:
      ret['ltrb_amodal'][k] = \
        bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
        bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
      ret['ltrb_amodal_mask'][k] = 1
      gt_det['ltrb_amodal'].append(bbox_amodal)
      ret['hm_out_amodal'][:, ct_int[1], ct_int[0]] = [bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1],
                                                       bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]]

    if 'nuscenes_att' in self.opt.heads:
      if ('attributes' in ann) and ann['attributes'] > 0:
        att = int(ann['attributes'] - 1)
        ret['nuscenes_att'][k][att] = 1
        ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
      gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    if 'velocity' in self.opt.heads:
      if ('velocity' in ann) and min(ann['velocity']) > -1000:
        ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
        ret['velocity_mask'][k] = 1
      gt_det['velocity'].append(ret['velocity'][k])

    if 'hps' in self.opt.heads:
      self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

    if 'rot' in self.opt.heads:
      self._add_rot(ret, ann, k, gt_det)

    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        ret['dep_mask'][k] = 1
        ret['dep'][k] = ann['depth'] * aug_s
        gt_det['dep'].append(ret['dep'][k])
      else:
        gt_det['dep'].append(2)

    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        ret['dim_mask'][k] = 1
        ret['dim'][k] = ann['dim']
        gt_det['dim'].append(ret['dim'][k])
      else:
        gt_det['dim'].append([1,1,1])
    
    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        amodel_center = affine_transform(ann['amodel_center'], trans_output)
        ret['amodel_offset_mask'][k] = 1
        ret['amodel_offset'][k] = amodel_center - ct_int
        gt_det['amodel_offset'].append(ret['amodel_offset'][k])
      else:
        gt_det['amodel_offset'].append([0, 0])
    

  def _add_hps(self, ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w):
    num_joints = self.num_joints
    pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3) \
        if 'keypoints' in ann else np.zeros((self.num_joints, 3), np.float32)
    if self.opt.simple_radius > 0:
      hp_radius = int(simple_radius(h, w, min_overlap=self.opt.simple_radius))
    else:
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))

    for j in range(num_joints):
      pts[j, :2] = affine_transform(pts[j, :2], trans_output)
      if pts[j, 2] > 0:
        if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
          pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h:
          ret['hps'][k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
          ret['hps_mask'][k, j * 2: j * 2 + 2] = 1
          pt_int = pts[j, :2].astype(np.int32)
          ret['hp_offset'][k * num_joints + j] = pts[j, :2] - pt_int
          ret['hp_ind'][k * num_joints + j] = \
            pt_int[1] * self.opt.output_w + pt_int[0]
          ret['hp_offset_mask'][k * num_joints + j] = 1
          ret['hm_hp_mask'][k * num_joints + j] = 1
          ret['joint'][k * num_joints + j] = j
          draw_umich_gaussian(
            ret['hm_hp'][j], pt_int, hp_radius)
          if pts[j, 2] == 1:
            ret['hm_hp'][j, pt_int[1], pt_int[0]] = self.ignore_val
            ret['hp_offset_mask'][k * num_joints + j] = 0
            ret['hm_hp_mask'][k * num_joints + j] = 0
        else:
          pts[j, :2] *= 0
      else:
        pts[j, :2] *= 0
        self._ignore_region(
          ret['hm_hp'][j, int(bbox[1]): int(bbox[3]) + 1, 
                          int(bbox[0]): int(bbox[2]) + 1])
    gt_det['hps'].append(pts[:, :2].reshape(num_joints * 2))

  def _add_rot(self, ret, ann, k, gt_det):
    if 'alpha' in ann:
      ret['rot_mask'][k] = 1
      alpha = ann['alpha']
      if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
        ret['rotbin'][k, 0] = 1
        ret['rotres'][k, 0] = alpha - (-0.5 * np.pi)    
      if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
        ret['rotbin'][k, 1] = 1
        ret['rotres'][k, 1] = alpha - (0.5 * np.pi)
      gt_det['rot'].append(self._alpha_to_8(ann['alpha']))
    else:
      gt_det['rot'].append(self._alpha_to_8(0))
    
  def _alpha_to_8(self, alpha):
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
  
  def _format_gt_det(self, gt_det):
    if (len(gt_det['scores']) == 0):
      gt_det = {'bboxes': np.array([[0,0,1,1]], dtype=np.float32), 
                'scores': np.array([1], dtype=np.float32), 
                'clses': np.array([0], dtype=np.float32),
                'cts': np.array([[0, 0]], dtype=np.float32),
                'pre_cts': np.array([[0, 0]], dtype=np.float32),
                'tracking': np.array([[0, 0]], dtype=np.float32),
                'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                'hps': np.zeros((1, 17, 2), dtype=np.float32),}
    gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
    return gt_det

  def fake_video_data(self):
    self.coco.dataset['videos'] = []
    for i in range(len(self.coco.dataset['images'])):
      img_id = self.coco.dataset['images'][i]['id']
      self.coco.dataset['images'][i]['video_id'] = img_id
      self.coco.dataset['images'][i]['frame_id'] = 1
      self.coco.dataset['videos'].append({'id': img_id})
    
    if not ('annotations' in self.coco.dataset):
      return

    for i in range(len(self.coco.dataset['annotations'])):
      self.coco.dataset['annotations'][i]['track_id'] = i + 1

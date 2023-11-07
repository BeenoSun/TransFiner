from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from progress.bar import Bar
import time
import torch
# print(torch.cuda.is_available())
# print(torch.cuda.is_initialized())
import math
#from trainer import second_round, second_round_origin
from model.model import create_model, load_model
from model.decode import generic_decode
from model.utils import flip_tensor, flip_lr_off, flip_lr, _nms
from utils.image import get_affine_transform, affine_transform
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.post_process import generic_post_process
from utils.debugger import Debugger
from utils.tracker import Tracker
from dataset.dataset_factory import get_dataset
from torch.cuda.amp import autocast
from post_transfiner.transfiner import build_eval
import post_transfiner.utils.misc as utils
from opts import opts
opt = opts().parse()
class Detector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')

    print('Creating model...')
    self.model = create_model(
      opt.arch, opt.heads, opt.head_conv, opt=opt)
    #model_ori = torch.load(
     # '/home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/paper_original_custom_dataset_amp_gradaccu_bsz16/model_70.pth')
    #self.model.load_state_dict(model_ori['state_dict'], strict=False)
    self.model_second, _, self.postprocessors = build_eval(opt)
    self.model, self.model_second = load_model(self.model, opt.load_model, self.model_second, opt)
    self.model, self.model_second = \
      self.model.to(opt.device), self.model_second.to(opt.device)
    self.model.eval()
    self.model_second.eval()
    #self.model_fusion.eval()

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
    self.area = torch.tensor(0, device=opt.device)
    self.num_re = 0
    self.val_r = 0
    self.total_frac = 0

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

    detections, detections_ori, detections_potential = [], [], []

    # for multi-scale testing
    for scale in self.opt.test_scales:
      scale_start_time = time.time()
      if not pre_processed:
        # not prefetch testing or demo
        images, meta = self.pre_process(image, scale, meta)
      else:
        # prefetch testing
        images = pre_processed_images['images'][scale][0]
        ori_images = pre_processed_images['ori_images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        mask = pre_processed_images['mask']
        if 'pre_dets' in pre_processed_images['meta']:
          meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
        if 'cur_dets' in pre_processed_images['meta']:
          meta['cur_dets'] = pre_processed_images['meta']['cur_dets']

      images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)
      ori_images = ori_images.to(self.opt.device, non_blocking=self.opt.non_block_test)

      # initializing tracker
      pre_hms, hms, pre_inds = None, None, None
      if self.opt.tracking:
        # initialize the first frame
        if self.pre_images is None:
          print('Initialize tracking!')
          self.pre_images = images
          self.ori_pre_images = ori_images
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
            self.tracker_temp.tracks, meta, with_hm=not self.opt.zero_pre_hm)

      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time

      # run the network
      # output: the output feature maps, only used for visualizing
      # dets: output tensors after extracting peaks
      output, dets, dets_ori, forward_time = self.process(
        images, self.pre_images, pre_hms, pre_inds=pre_inds, return_time=True, meta=meta, mask=mask,
        gt_box=image_or_path_or_tensor['bbox_amodal_gt'], ori_images=ori_images, ori_pre_images=self.ori_pre_images)
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time

      # convert the cropped and 4x downsampled output coordinate system
      # back to the input image coordinate system
      result = self.post_process(dets, meta, scale)
      result_ori = self.post_process(dets_ori, meta, scale, ori=True)
      #result_potential = self.post_process(dets_ori, meta, scale, potential=True)
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(result)
      detections_ori.append(result_ori)
      #detections_potential.append(result_potential)
      if self.opt.debug >= 2:
        self.debug(
          self.debugger, images, result, output, scale,
          pre_images=self.pre_images if not self.opt.no_pre_img else None,
          pre_hms=pre_hms)

    # merge multi-scale testing results
    results = self.merge_outputs(detections)
    results_ori = self.merge_outputs(detections_ori)
    #result_potential = self.merge_outputs(detections_potential, potential=True)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time

    if self.opt.tracking:
      # public detection mode in MOT challenge
      public_det = meta['cur_dets'] if self.opt.public_det else None
      # add tracking id to results
      results = self.tracker.step(results, public_det, ori=False)
      results_ori = self.tracker_temp.step(results_ori, public_det)
      self.pre_images = images
      self.ori_pre_images = ori_images

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

    padding_mask = np.ones_like(resized_image)
    inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
    padding_mask = cv2.warpAffine(padding_mask, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
    padding_mask = padding_mask[:, :, 0]
    padding_mask[padding_mask > 0] = 1

    # addition for higres
    ori_trans_input = get_affine_transform(c, s, 0, [opt.ori_input_w, opt.ori_input_h])
    ori_inp_image = cv2.warpAffine(resized_image, ori_trans_input, (opt.ori_input_w, opt.ori_input_h), flags=cv2.INTER_LINEAR)
    ori_inp_image = ((ori_inp_image / 255. - self.mean) / self.std).astype(np.float32)
    ori_images = ori_inp_image.transpose(2, 0, 1).reshape(1, 3, opt.ori_input_h, opt.ori_input_w)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
      ori_images = np.concatenate((ori_images, ori_images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    ori_images = torch.from_numpy(ori_images)
    padding_mask = torch.from_numpy(padding_mask)
    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
      if 'calib' in input_meta else \
      self._get_default_calib(width, height)}
    meta.update({'c': c, 's': s, 'height': height, 'width': width,
                 'out_height': out_height, 'out_width': out_width,
                 'inp_height': inp_height, 'inp_width': inp_width,
                 'trans_input': trans_input, 'trans_output': trans_output,
                 'ori_trans_input': ori_trans_input})
    if 'pre_dets' in input_meta:
      meta['pre_dets'] = input_meta['pre_dets']
    if 'cur_dets' in input_meta:
      meta['cur_dets'] = input_meta['cur_dets']
    return images, ori_images, 1 - padding_mask, meta

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
    trans_input, trans_output = meta['ori_trans_input'], meta['trans_output']
    inp_width, inp_height = opt.ori_input_w, opt.ori_input_h
    out_width, out_height = meta['out_width'], meta['out_height']
    ori_trans_input = meta['ori_trans_input']
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
              pre_inds=None, return_time=False, mask=None, meta=None, gt_box=None,
              ori_images=None, ori_pre_images=None):
    with torch.no_grad():
      torch.cuda.synchronize()
      with autocast(enabled=self.opt.withamp):
        output = self.model(ori_images, pre_img=ori_pre_images, pre_hm=pre_hms)
        output[0] = self._sigmoid_output(output[0])
        batch = {}
        #hm, hm_down = self.draw_hm(self.opt, meta, output, down_=True)
        #self.test_suspatch(self.opt, hm_down, output[0]['wh'], batch)  # program test only
        #generate_patch_gt(batch, hm_down, output[0]['ltrb_amodal'])
        # negloss_fn_gt, negloss_fp_gt = hm_corr_genera(opt, hm_down, batch, outputs_copy[0]['wh'])
        samples = utils.NestedTensor(images, mask)
        samples = samples.to(output[0]['hm'].device)
        pre_samples = utils.NestedTensor(pre_images, mask)
        pre_samples = pre_samples.to(output[0]['hm'].device)
        batch['tracker_out'] = output[0]
        batch['hm_out'] = output[0]['hm']
        #batch['hm_cat_withpatch'] = hm.detach().clone()
        batch['pad_mask'] = mask
        result_refine, patch_bbox = self.model_second([samples, pre_samples, batch])
        output[0].update({'result_refine': result_refine,
                          'patch_bbox': patch_bbox})
        # addition
        #output[0]['hm'] = pre_hms
        #output[0]['hm'] = (output[0]['hm'] + 1.) * hm_corr_hat - 1.

      output[0].update({'pre_inds': pre_inds})
      if self.opt.flip_test:
        output[0] = self._flip_output(output[0])
      torch.cuda.synchronize()
      forward_time = time.time()
      dets, dets_ori = generic_decode(output[0], K=self.opt.K, opt=self.opt, gt_box=gt_box)
      torch.cuda.synchronize()
      for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()
        dets_ori[k] = dets_ori[k].detach().cpu().numpy()
    if return_time:
      return output, dets, dets_ori, forward_time
    else:
      return output, dets, dets_ori


  def draw_hm(self, opt, meta, out, down_=False):
    meta_copy = meta
    #for nam in meta:
     # meta_copy[nam] = meta[nam].detach().clone()
    if down_:
      hms_d = np.zeros((out[0]['hm'].shape[0], 1, opt.output_h, opt.output_w), dtype=np.float32)
    hms_ = np.zeros((out[0]['hm'].shape[0], 1, opt.input_h, opt.input_w), dtype=np.float32)
    dets_ = generic_decode(out[0], K=opt.K, opt=opt)
    for k in dets_:
      dets_[k] = dets_[k].detach().cpu().numpy()
    for na in meta_copy:
      meta_copy[na] = meta_copy[na]
      b_ = 0
      det, met = {}, {}
      for k in dets_:
        det[k] = dets_[k][b_][np.newaxis]
      for na in meta_copy:
        met[na] = meta_copy[na]
      met['calib'] = None
      result = generic_post_process(
        opt, det, [met['c']], [met['s']],
        met['out_height'], met['out_width'], opt.num_classes,
        [met['calib']], met['height'], met['width'])
      detections = []
      detections.append(result[0])
      results = self.merge_outputs(detections)
      hms__, _ = self._get_additional_inputs(results, met)
      if down_:
        hms_d_, _ = self._get_additional_inputs(results, met, down=down_)
        hms_d[b_] = hms_d_[0].cpu().numpy()
      hms_[b_] = hms__[0].cpu().numpy()
    if down_:
      return torch.from_numpy(hms_).to(out[0]['hm'].device), torch.from_numpy(hms_d).to(out[0]['hm'].device)
    return torch.from_numpy(hms_).to(out[0]['hm'].device)


  def post_process(self, dets, meta, scale=1, potential=False, ori=False):
    dets = generic_post_process(
      self.opt, dets, [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'], self.opt.num_classes,
      [meta['calib']], meta['height'], meta['width'], potential=potential, ori=ori)
    self.this_calib = meta['calib']

    if scale != 1:
      for i in range(len(dets[0])):
        for k in ['bbox', 'hps']:
          if k in dets[0][i]:
            dets[0][i][k] = (np.array(
              dets[0][i][k], np.float32) / scale).tolist()
    return dets[0]

  def merge_outputs(self, detections, potential=False):
    assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
    results = []
    for i in range(len(detections[0])):
      if detections[0][i]['score'] > (self.opt.out_thresh if not potential else 0.01):
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
    self.tracker_temp.reset()
    self.pre_images = None
    self.pre_image_ori = None

  def test_suspatch(self, opt, hm_down, oupu_wh, bac):

    patch_gt = torch.zeros(hm_down.shape)
    for i_, pa in enumerate(bac['patch_targets']):
        out_hm = torch.zeros(hm_down.shape[1:])
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

    patch_pred = torch.zeros(hm_down.shape)
    for i_ in range(bac['patch_box_pred'].shape[0]):
        out_hm = torch.zeros(hm_down.shape[1:])
        if (bac['patch_box_pred'][i_] - torch.tensor([0.5, 0.5, 1., 1.], device=oupu_wh.device)).sum() == 0:
            patch_pred[i_] = out_hm
            continue

        for i in range(opt.num_patch_val):
            box = (torch.tensor([opt.input_w // 4, opt.input_h // 4, opt.input_w // 4, opt.input_h // 4], device=oupu_wh.device) *
                   bac['patch_box_pred'][i_][
                       torch.sort(bac['patch_box_cls'][i_, :, 1:].max(1).values, dim=0, descending=True).indices[i]]).squeeze()

            box[:2] = box[:2] - box[2:] / 2
            box[2:] = box[:2] + box[2:]
            box = box.detach().cpu().numpy()
            le = int(np.clip(box[0], 0, opt.input_w // 4 - 1))
            to = int(np.clip(box[1], 0, opt.input_h // 4 - 1))
            ri = int(np.clip(box[2], 0, opt.input_w // 4 - 1))
            bo = int(np.clip(box[3], 0, opt.input_h // 4 - 1))
            out_hm[0, to: (bo + 1), le: (ri + 1)] = np.maximum(out_hm[0, to: (bo + 1), le: (ri + 1)].cpu(), i+1)
            patch_pred[i_] = out_hm

    bac['patch_area_pred'] = patch_pred
    bac['patch_area'] = patch_gt
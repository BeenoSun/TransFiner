from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import copy

from utils.image import get_affine_transform, affine_transform
from opts import opts
opt = opts().parse()
if (opt.load_model.split('/')[-2] == 'mot17_half') or (opt.load_model.split('/')[-3] == 'mot17_half'):
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory
from detector_double import Detector
import pycocotools.coco as coco

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.get_default_calib = dataset.get_default_calib
    self.opt = opt
    self.coco = coco.COCO(dataset.img_dir[:-5] + '/annotations/'+'test.json') # addition

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    _, anns, _, _ = self._load_data(index)
    #print(torch.tensor([ann['iscrowd'] > 0 for ann in anns]).sum())
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, ori_images, meta = {}, {}, {}
    for scale in opt.test_scales:
      input_meta = {}
      calib = img_info['calib'] if 'calib' in img_info \
        else self.get_default_calib(image.shape[1], image.shape[0])
      input_meta['calib'] = calib
      images[scale], ori_images[scale], mask, meta[scale] = self.pre_process_func(
        image, scale, input_meta)
      #bbox_amodal = torch.stack([torch.from_numpy(self._get_bbox_output(ann['bbox'], meta[scale]['trans_output'])[-1]) for ann in anns])
    ret = {'images': images, 'ori_images': ori_images,'image': image, 'meta': meta, 'mask': mask, 'bbox_amodal_gt': torch.zeros(4,4).to(device=mask.device),
           'bbox_amodal_or': anns}
    if 'frame_id' in img_info and img_info['frame_id'] == 1:
      ret['is_first_frame'] = 1
      ret['video_id'] = img_info['video_id']
    return img_id, ret

  def __len__(self):
    return len(self.images)

  def _load_data(self, index):
    coco = self.coco
    img_dir = self.img_dir
    img_id = self.images[index]

    img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

    return img, anns, img_info, img_path

  def _load_image_anns(self, img_id, coco, img_dir):
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  def _get_bbox_output(self, bbox, trans_output, height=None, width=None):
    bbox = self._coco_box_to_bbox(bbox).copy() # topx, topy, botx, boty

    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    for t in range(4):
      rect[t] =  affine_transform(rect[t], trans_output)
    bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
    bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

    bbox_amodal = copy.deepcopy(bbox)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    return bbox, bbox_amodal

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

def prefetch_test(opt):
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  if opt.load_results != '':
    load_results = json.load(open(opt.load_results, 'r'))
    for img_id in load_results:
      for k in range(len(load_results[img_id])):
        if load_results[img_id][k]['class'] - 1 in opt.ignore_loaded_cats:
          load_results[img_id][k]['score'] = -1
  else:
    load_results = {}

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  if opt.use_loaded_results:
    for img_id in data_loader.dataset.images:
      results[img_id] = load_results['{}'.format(img_id)]
    num_iters = 0
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    if ind >= num_iters:
      break

    # test seperate video sequence
    '''vid_spe = 16
    global vid
    vid = pre_processed_images['video_id'] if 'video_id' in pre_processed_images else vid
    if vid != vid_spe:
      print('skip img', img_id)
      continue'''

    if opt.tracking and ('is_first_frame' in pre_processed_images):
      if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
        pre_processed_images['meta']['pre_dets'] = \
          load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
      else:
        print()
        '''if vid == 1:
          opt.refine_thresh = 0.6'''
        print('No pre_dets for', int(img_id.numpy().astype(np.int32)[0]), 
          '. Use empty initialization.')
        pre_processed_images['meta']['pre_dets'] = []
      detector.reset_tracking()
      print('Start tracking video', int(pre_processed_images['video_id']))
    if opt.public_det:
      if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
        pre_processed_images['meta']['cur_dets'] = \
          load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
      else:
        print('No cur_dets for', int(img_id.numpy().astype(np.int32)[0]))
        pre_processed_images['meta']['cur_dets'] = []
    
    ret = detector.run(pre_processed_images)
    results[int(img_id.numpy().astype(np.int32)[0])] = ret['results']
    
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    if opt.print_iter > 0:
      if ind % opt.print_iter == 0:
        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
    else:
      bar.next()
  bar.finish()

  if opt.save_results:
    print('saving results to', opt.save_dir + '/save_results_{}{}.json'.format(
      opt.test_dataset, opt.dataset_version))
    json.dump(_to_list(copy.deepcopy(results)), 
              open(opt.save_dir + '/save_results_{}{}.json'.format(
                opt.test_dataset, opt.dataset_version), 'w'))
  dataset.run_eval(results, opt.save_dir)

def test(opt):
  #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  if opt.load_results != '': # load results in json
    load_results = json.load(open(opt.load_results, 'r'))

  results = {}
  num_iters = len(dataset) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]

    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    input_meta = {}
    if 'calib' in img_info:
      input_meta['calib'] = img_info['calib']
    if (opt.tracking and ('frame_id' in img_info) and img_info['frame_id'] == 1):
      detector.reset_tracking()
      input_meta['pre_dets'] = load_results[img_id]

    ret = detector.run(img_path, input_meta)    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  if opt.save_results:
    print('saving results to', opt.save_dir + '/save_results_{}{}.json'.format(
      opt.test_dataset, opt.dataset_version))
    json.dump(_to_list(copy.deepcopy(results)), 
              open(opt.save_dir + '/save_results_{}{}.json'.format(
                opt.test_dataset, opt.dataset_version), 'w'))
  dataset.run_eval(results, opt.save_dir)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().parse()
  # from multiprocessing import set_start_method as _set_start_method
  # _set_start_method('spawn')
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)

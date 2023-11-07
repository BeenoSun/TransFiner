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
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory
from detector_double import Detector
import pycocotools.coco as coco


def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results


Dataset = dataset_factory[opt.test_dataset]
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
print(opt)
dataset = Dataset(opt, 'val')
f = open('/home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/save_results_mot17halfval.json')
results = json.load(f)
#results = []#json.load('/home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/save_results_mot17halfval.json')
dataset.run_eval(results, opt.save_dir)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import post_transfiner.utils.misc as utils
import time
import torch
import numpy as np
from progress.bar import Bar
from model.data_parallel import DataParallel
from utils.utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from model.losses import FastFocalLoss, RegWeightedL1Loss, HMCorrLoss
from model.losses import BinRotLoss, WeightedBCELoss
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
import math
from utils.post_process import generic_post_process
from utils.image import draw_umich_gaussian, gaussian_radius
from post_transfiner.utils.utils import hm_corr_genera, generate_patch_gt, test_suspatch
from detector_double import Detector
#torch.autograd.set_detect_anomaly(True)
from opts import opts
opt = opts().parse()

class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
      super(GenericLoss, self).__init__()
      self.crit = FastFocalLoss(opt=opt)
      self.crit_reg = RegWeightedL1Loss()
      if 'rot' in opt.heads:
          self.crit_rot = BinRotLoss()
      if 'nuscenes_att' in opt.heads:
          self.crit_nuscenes_att = WeightedBCELoss()
      self.opt = opt

  def _sigmoid_output(self, output):
      if 'hm' in output:
          output['hm'] = _sigmoid(output['hm'])
      if 'hm_hp' in output:
          output['hm_hp'] = _sigmoid(output['hm_hp'])
      if 'dep' in output:
          output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      return output

  def forward(self, outputs, batch):
      opt = self.opt
      losses = {head: 0 for head in opt.heads}

      for s in range(opt.num_stacks):
          output = outputs[s]
          # output = self._sigmoid_output(output)

          if 'hm' in output:
              losses['hm'] += self.crit(
                  output['hm'], batch['hm'], batch['ind'],
                  batch['mask'], batch['cat']) / opt.num_stacks

          regression_heads = [
              'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
              'dep', 'dim', 'amodel_offset', 'velocity']

          for head in regression_heads:
              if head in output:
                  losses[head] += self.crit_reg(
                      output[head], batch[head + '_mask'],
                      batch['ind'], batch[head]) / opt.num_stacks

          if 'hm_hp' in output:
              losses['hm_hp'] += self.crit(
                  output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                  batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
              if 'hp_offset' in output:
                  losses['hp_offset'] += self.crit_reg(
                      output['hp_offset'], batch['hp_offset_mask'],
                      batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

          if 'rot' in output:
              losses['rot'] += self.crit_rot(
                  output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                  batch['rotres']) / opt.num_stacks

          if 'nuscenes_att' in output:
              losses['nuscenes_att'] += self.crit_nuscenes_att(
                  output['nuscenes_att'], batch['nuscenes_att_mask'],
                  batch['ind'], batch['nuscenes_att']) / opt.num_stacks

      losses['tot'] = 0
      for head in opt.heads:
          losses['tot'] += opt.weights[head] * losses[head]

      return losses['tot'], losses


class ModleWithLoss(torch.nn.Module):
  def __init__(self, opt, model, model_second, loss, loss_second):
      super(ModleWithLoss, self).__init__()
      self.opt = opt
      self.model = model
      #self.model.eval()
      self.model_second = model_second
      self.loss = loss
      self.loss_second = loss_second
      self.detector = Detector(opt)

  @autocast(enabled=opt.withamp)
  def forward(self, batch):
      with torch.no_grad():
        outputs = self.model(batch['ori_inp_image'], pre_img=batch['ori_inp_image_pre'], pre_hm=batch['pre_hm'])
        if 'hm' in outputs[0]:
          outputs[0]['hm'] = _sigmoid(outputs[0]['hm'])
      #   stage 2：BiConvLSTM is employed to generate the hm_corr_hat
      outputs_copy = [{}]
      for stac in range(len(outputs)):
        for nam in outputs[stac]:
          outputs_copy[stac][nam] = outputs[stac][nam].detach().clone()

      #'''
      # transfiner implementation by beeno, v1, jan 2022
      #hm, hm_down = draw_hm(self, self.opt, batch['cur_meta'], outputs_copy, down_=True)
      #generate_patch_gt(batch, hm_down, outputs_copy[0]['ltrb_amodal'])
      #negloss_fn_gt, negloss_fp_gt = hm_corr_genera(opt, hm_down, batch, outputs_copy[0]['wh'])
      samples = utils.NestedTensor(batch['image'], batch['pad_mask'])
      samples = samples.to(outputs[0]['hm'].device)
      pre_samples = utils.NestedTensor(batch['pre_img'], batch['pre_pad_mask'])
      pre_samples = pre_samples.to(outputs[0]['hm'].device)
      batch['tracker_out'] = outputs_copy[0]
      batch['hm_out'] = outputs_copy[0]['hm']
      #batch['hm_cat_withpatch'] = hm.detach().clone()
      outputs = self.model_second([samples, pre_samples, batch])
      loss_dict = self.loss_second(outputs, batch['refine_targets'], batch)
      #test_suspatch(opt, hm_down, outputs_copy[0]['wh'], batch) # program test only
      weight_dict = self.loss_second.weight_dict
      losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

      #'''
      loss_stats = {}
      loss_stats['loss_second'] = losses
      loss_stats['refine_ce'] = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if ('ce' in k and 'patch' not in k and 'pre' not in k)) \
                                if not opt.transpatch_trainonly else torch.tensor(0., device=losses.device)
      loss_stats['refine_coord'] = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if ((('box' in k) or ('iou' in k)) and ('patch' not in k) and ('pre' not in k))) \
                                if not opt.transpatch_trainonly else torch.tensor(0., device=losses.device)
      loss_stats['pre_coord'] = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if ((('box' in k) or ('iou' in k)) and ('patch' not in k) and ('pre' in k))) \
                                if not opt.transpatch_trainonly else torch.tensor(0., device=losses.device)
      loss_stats['patch_ce'] = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if ('ce' in k and 'patch' in k)) \
                                if not opt.transformer_trainonly else torch.tensor(0., device=losses.device)
      loss_stats['patch_coord'] = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if ((('box' in k) or ('iou' in k)) and ('patch' in k))) \
                                if not opt.transformer_trainonly else torch.tensor(0., device=losses.device)

      return losses, loss_stats

def draw_hm(self, opt, meta, out, down_=False):
    meta_copy = {}
    for nam in meta:
        meta_copy[nam] = meta[nam].detach().clone()
    if down_:
        hms_d = np.zeros((out[0]['hm'].shape[0], 1, opt.output_h, opt.output_w), dtype=np.float32)
    hms_ = np.zeros((out[0]['hm'].shape[0], 1, opt.input_h, opt.input_w), dtype=np.float32)
    dets_ = generic_decode(out[0], K=opt.K, opt=opt)
    for k in dets_:
        dets_[k] = dets_[k].detach().cpu().numpy()
    for na in meta_copy:
        meta_copy[na] = meta_copy[na].cpu().numpy()
    for b_ in range(out[0]['hm'].shape[0]):
        det, met = {}, {}
        for k in dets_:
            det[k] = dets_[k][b_][np.newaxis]
        for na in meta_copy:
            met[na] = meta_copy[na][b_]
        met['calib'] = None
        result = generic_post_process(
            opt, det, [met['c']], [met['s']],
            met['out_height'], met['out_width'], opt.num_classes,
            [met['calib']], met['height'], met['width'])
        detections = []
        detections.append(result[0])
        results = self.detector.merge_outputs(detections)
        hms__, _ = self.detector._get_additional_inputs(results, met)
        if down_:
            hms_d_, _ = self.detector._get_additional_inputs(results, met, down=down_)
            hms_d[b_] = hms_d_[0].cpu().numpy()
        hms_[b_] = hms__[0].cpu().numpy()
    if down_:
        return torch.from_numpy(hms_).to(out[0]['hm'].device), torch.from_numpy(hms_d).to(out[0]['hm'].device)
    return torch.from_numpy(hms_).to(out[0]['hm'].device)


class Trainer(object):
  def __init__(
          self, opt, model, optimizer, model_second, optimizer_second, criterion_second):
      self.opt = opt
      self.optimizer = optimizer
      self.optimizer_second = optimizer_second
      self.loss_stats, self.loss, _ = self._get_losses(opt)
      self.loss_second = criterion_second
      self.loss_stats = []
      self.loss_stats.append('loss_second')
      self.loss_stats.append('refine_ce')
      self.loss_stats.append('refine_coord')
      self.loss_stats.append('pre_coord')
      self.loss_stats.append('patch_ce')
      self.loss_stats.append('patch_coord')
      #self.loss_stats.append('loss_fusion')
      self.model_with_loss = ModleWithLoss(opt, model, model_second,\
                                           self.loss, self.loss_second)

  def set_device(self, gpus, chunk_sizes, device):
      if len(gpus) > 1:
          self.model_with_loss = DataParallel(
              self.model_with_loss, device_ids=gpus,
              chunk_sizes=chunk_sizes).to(device)
      else:
          self.model_with_loss = self.model_with_loss.to(device)

      for state in self.optimizer.state.values():
          for k, v in state.items():
              if isinstance(v, torch.Tensor):
                  state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader, scaler, scaler_second):
      model_with_loss = self.model_with_loss
      if phase == 'train':
          model_with_loss.train()
      else:
          if len(self.opt.gpus) > 1:
              model_with_loss = self.model_with_loss.module
          model_with_loss.eval()
          torch.cuda.empty_cache()

      opt = self.opt
      results = {}
      data_time, batch_time = AverageMeter(), AverageMeter()
      avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
      num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
      bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
      end = time.time()
      for iter_id, batch in enumerate(data_loader):
          if iter_id >= num_iters:
              break
          data_time.update(time.time() - end)

          for k in batch:
              if (k != 'meta') and (k != 'cur_meta') and (k != 'pre_meta') and (k != 'inp_ori') and (k != 'anns_ori') and (k != 'test_img'):
                  batch[k] = batch[k].to(device=opt.device, non_blocking=True)

          if phase == 'train':
              with autocast(enabled=self.opt.withamp):
                loss_second, loss_stats = model_with_loss(batch)
                loss_second = loss_second.mean()

                #loss_cor = loss_cor.mean()
                #loss_res = loss_res.mean()
                # gradaccu
                loss_second = loss_second / opt.accum_iter
                #loss_second = loss_cor + loss_res
          else:
              with torch.no_grad():
                  with autocast(enabled=self.opt.withamp):
                    loss_second, loss_stats = model_with_loss(batch)
                    loss_second = loss_second.mean()

          if phase == 'train':
              #if epoch > 2:

              if self.opt.withamp:
                  # scaler.scale(loss).backward()
                  i = 1 # useless
                  scaler_second.scale(loss_second).backward()
                  if ((iter_id + 1) % opt.accum_iter == 0) or ((iter_id + 1) == len(data_loader)):
                    scaler_second.step(self.optimizer_second)
                    scaler_second.update()
                    self.optimizer_second.zero_grad()


                  #scaler_fusion.scale(loss_fus).backward()
                  # loss.backward()
                  # loss_second.backward()
                  # if ((iter_id + 1) % opt.accum_iter == 0) or ((iter_id + 1) == len(data_loader)):
                  # if epoch > 2:
                  # scaler.step(self.optimizer)
                  # scaler.update()
                  # self.optimizer.zero_grad()
                  #scaler_fusion.step(self.optimizer_fusion)
                  #scaler_fusion.update()
                  #self.optimizer_fusion.zero_grad()
              else:
                #with torch.autograd.detect_anomaly():
                loss_second.backward()
                if ((iter_id + 1) % opt.accum_iter == 0) or ((iter_id + 1) == len(data_loader)):
                    self.optimizer_second.step()
                    self.optimizer_second.zero_grad()


          batch_time.update(time.time() - end)
          end = time.time()

          Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
              epoch, iter_id, num_iters, phase=phase,
              total=bar.elapsed_td, eta=bar.eta_td)
          for l in avg_loss_stats:
              avg_loss_stats[l].update(
                  loss_stats[l].mean().item(), batch['image'].size(0))
              Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
          Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
          if opt.print_iter > 0:  # If not using progress bar
              if iter_id % opt.print_iter == 0:
                  print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
          else:
              bar.next()

          if opt.debug > 0:
              self.debug(batch, iter_id, dataset=data_loader.dataset)

          del loss_second, loss_stats

      bar.finish()
      ret = {k: v.avg for k, v in avg_loss_stats.items()}
      ret['time'] = bar.elapsed_td.total_seconds() / 60.
      return ret, results

  def _get_losses(self, opt):
      loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
                    'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
                    'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
      loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
      loss = GenericLoss(opt)
      loss_second = HMCorrLoss(opt)
      return loss_states, loss, loss_second

  def debug(self, batch, output, iter_id, dataset):
      opt = self.opt
      if 'pre_hm' in batch:
          output.update({'pre_hm': batch['pre_hm']})
      dets = generic_decode(output, K=opt.K, opt=opt)
      for k in dets:
          dets[k] = dets[k].detach().cpu().numpy()
      dets_gt = batch['meta']['gt_det']
      for i in range(1):
          debugger = Debugger(opt=opt, dataset=dataset)
          img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
          img = np.clip(((
                                 img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
          pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
          gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
          debugger.add_blend_img(img, pred, 'pred_hm')
          debugger.add_blend_img(img, gt, 'gt_hm')

          if 'pre_img' in batch:
              pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
              pre_img = np.clip(((
                                         pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
              debugger.add_img(pre_img, 'pre_img_pred')
              debugger.add_img(pre_img, 'pre_img_gt')
              if 'pre_hm' in batch:
                  pre_hm = debugger.gen_colormap(
                      batch['pre_hm'][i].detach().cpu().numpy())
                  debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

          debugger.add_img(img, img_id='out_pred')
          if 'ltrb_amodal' in opt.heads:
              debugger.add_img(img, img_id='out_pred_amodal')
              debugger.add_img(img, img_id='out_gt_amodal')

          # Predictions
          for k in range(len(dets['scores'][i])):
              if dets['scores'][i, k] > opt.vis_thresh:
                  debugger.add_coco_bbox(
                      dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
                      dets['scores'][i, k], img_id='out_pred')

                  if 'ltrb_amodal' in opt.heads:
                      debugger.add_coco_bbox(
                          dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
                          dets['scores'][i, k], img_id='out_pred_amodal')

                  if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
                      debugger.add_coco_hp(
                          dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

                  if 'tracking' in opt.heads:
                      debugger.add_arrow(
                          dets['cts'][i][k] * opt.down_ratio,
                          dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
                      debugger.add_arrow(
                          dets['cts'][i][k] * opt.down_ratio,
                          dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

          # Ground truth
          debugger.add_img(img, img_id='out_gt')
          for k in range(len(dets_gt['scores'][i])):
              if dets_gt['scores'][i][k] > opt.vis_thresh:
                  debugger.add_coco_bbox(
                      dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
                      dets_gt['scores'][i][k], img_id='out_gt')

                  if 'ltrb_amodal' in opt.heads:
                      debugger.add_coco_bbox(
                          dets_gt['bboxes_amodal'][i, k] * opt.down_ratio,
                          dets_gt['clses'][i, k],
                          dets_gt['scores'][i, k], img_id='out_gt_amodal')

                  if 'hps' in opt.heads and \
                          (int(dets['clses'][i, k]) == 0):
                      debugger.add_coco_hp(
                          dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

                  if 'tracking' in opt.heads:
                      debugger.add_arrow(
                          dets_gt['cts'][i][k] * opt.down_ratio,
                          dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
                      debugger.add_arrow(
                          dets_gt['cts'][i][k] * opt.down_ratio,
                          dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

          if 'hm_hp' in opt.heads:
              pred = debugger.gen_colormap_hp(
                  output['hm_hp'][i].detach().cpu().numpy())
              gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
              debugger.add_blend_img(img, pred, 'pred_hmhp')
              debugger.add_blend_img(img, gt, 'gt_hmhp')

          if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
              dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
              calib = batch['meta']['calib'].detach().numpy() \
                  if 'calib' in batch['meta'] else None
              det_pred = generic_post_process(opt, dets,
                                              batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                              output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                              calib)
              det_gt = generic_post_process(opt, dets_gt,
                                            batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                            output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                            calib)

              debugger.add_3d_detection(
                  batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                  det_pred[i], calib[i],
                  vis_thresh=opt.vis_thresh, img_id='add_pred')
              debugger.add_3d_detection(
                  batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                  det_gt[i], calib[i],
                  vis_thresh=opt.vis_thresh, img_id='add_gt')
              debugger.add_bird_views(det_pred[i], det_gt[i],
                                      vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')

          if opt.debug == 4:
              debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
          else:
              debugger.show_all_imgs(pause=True)

  def val(self, epoch, data_loader, scaler=None, scaler_second=None):
      return self.run_epoch('val', epoch, data_loader, scaler, scaler_second)

  def train(self, epoch, data_loader, scaler, scaler_second):
      return self.run_epoch('train', epoch, data_loader, scaler, scaler_second)

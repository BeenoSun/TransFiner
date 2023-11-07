from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar

from model.data_parallel import DataParallel
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, RegWeightedL1Loss, HMCorrLoss
from model.losses import BinRotLoss, WeightedBCELoss
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import generic_post_process


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
      self.model_second = model_second
      self.loss = loss
      self.loss_second = loss_second

  def forward(self, batch):
      # step 1: using 'pre2_hm' to get the 'pre_hm' for further forward propagatation
      with torch.no_grad():
          #   stage 1：CenterTrack output hm, tracking, offset, wh
          # self.model(images, pre_images, pre_hms)
          outputs = self.model(batch['pre_img'], batch['pre2_img'], batch['pre2_hm'])
          if 'hm' in outputs[0]:
              outputs[0]['hm'] = _sigmoid(outputs[0]['hm'])
          #   stage 2：BiConvLSTM is employed to generate the hm_corr_hat
          inputs_second = second_round(self.opt, outputs)
          hm_corr_hat = self.model_second(inputs_second.to(outputs[0]['hm'].device))
          batch['pre_hm'] = ((outputs[0]['hm'] + 1.) * hm_corr_hat - 1.)

      # step2 : using the 'pre_hm' generated to predict 'hm' train the network
      #   stage 1：CenterTrack output hm, tracking, offset, wh
      outputs = self.model(batch['image'], batch['pre_img'], batch['pre_hm'])
      if 'hm' in outputs[0]:
          outputs[0]['hm'] = _sigmoid(outputs[0]['hm'])
      #   stage 2：BiConvLSTM is employed to generate the hm_corr_hat
      outputs_copy = [{}]
      for stac in range(len(outputs)):
          for nam in outputs[stac]:
              outputs_copy[stac][nam] = outputs[stac][nam].detach().clone()
      inputs_second = second_round(self.opt, outputs_copy)
      hm_corr_hat = self.model_second(inputs_second.to(outputs[0]['hm'].device))

      # calculate the loss for training
      hm_corr_gt = (batch['hm'] + 1.) / (outputs_copy[0]['hm'] + 1.)
      loss_second = self.loss_second(hm_corr_hat, hm_corr_gt, batch)
      loss, loss_stats = self.loss(outputs, batch)
      loss_stats['loss_second'] = loss_second

      return outputs[-1], loss, loss_second, loss_stats


def second_round(opt, outputs):
    '''
  input：
    outputs ： outputs from former tracker
  output:
    input_second ： inputs for the hm correlator
  '''
    outputs_ = [{}]
    for stac in range(len(outputs)):
        for nam in outputs[stac]:
            outputs_[stac][nam] = torch.transpose(outputs[stac][nam], 2, 3)
    x = torch.linspace(0, outputs_[0]['hm'].shape[2] - 1, outputs_[0]['hm'].shape[2]) \
        .unsqueeze(0).unsqueeze(1).unsqueeze(3).to(outputs_[0]['hm'].device)
    y = torch.linspace(0, outputs_[0]['hm'].shape[3] - 1, outputs_[0]['hm'].shape[3]) \
        .unsqueeze(0).unsqueeze(1).unsqueeze(2).to(outputs_[0]['hm'].device)
    z_len = min(outputs_[0]['hm'].shape[2], outputs_[0]['hm'].shape[3])
    z = torch.linspace(0, z_len - 1, z_len).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(outputs_[0]['hm'].device)
    xyz = opt.x_weight * x + opt.y_weight * y + opt.z_weight * z
    corr_z = torch.zeros(outputs_[0]['hm'].shape[0], z_len, 1, 1).to(outputs_[0]['hm'].device)
    for z_ in range(z_len):
        corr_z[:, z_, 0, 0] = outputs_[0]['hm'][:, 0, z_, :].sum(1) + \
                              outputs_[0]['hm'][:, 0, :, z_].sum(1)
    #   offset box
    corr_offsetx = outputs_[0]['reg'][:, 0, :, :].unsqueeze(1).sum(3).unsqueeze(3)
    corr_offsety = outputs_[0]['reg'][:, 1, :, :].unsqueeze(1).sum(2).unsqueeze(2)
    corr_offsetxyz = z_len * (opt.x_weight * corr_offsetx + \
                                                  opt.y_weight * corr_offsety + opt.z_weight * corr_z) \
                     / (xyz + z_len * opt.offset_denomin)
    #   wh box
    corr_whx = outputs_[0]['wh'][:, 0, :, :].unsqueeze(1).sum(3).unsqueeze(3)
    corr_why = outputs_[0]['wh'][:, 1, :, :].unsqueeze(1).sum(2).unsqueeze(2)
    corr_whxyz = z_len * (opt.x_weight * corr_whx + \
                                              opt.y_weight * corr_why + opt.z_weight * corr_z) \
                 / (xyz + z_len * opt.wh_denomin)
    #   tracking box
    corr_trackingx = outputs_[0]['tracking'][:, 0, :, :].unsqueeze(1).sum(3).unsqueeze(3)
    corr_trackingy = outputs_[0]['tracking'][:, 1, :, :].unsqueeze(1).sum(2).unsqueeze(2)
    corr_trackingxyz = z_len * (opt.x_weight * corr_trackingx + \
                                                    opt.y_weight * corr_trackingy + opt.z_weight * corr_z) \
                       / (xyz + z_len * opt.tracking_denomin)

    biconvlstm_seq_len = (2 * min(opt.output_h, opt.output_w) + max(opt.output_h, opt.output_w)) //  opt.stack_num
    input_second = torch.zeros(outputs_[0]['hm'].shape[0],
                               biconvlstm_seq_len,
                               opt.biconvlstm_input_dim * opt.stack_num,
                               opt.output_w,
                               opt.output_h)
    single_layer_input_shape = [outputs_[0]['hm'].shape[0],
                                opt.biconvlstm_input_dim * opt.stack_num,
                                opt.output_w,
                                opt.output_h]
    for len_ in range(biconvlstm_seq_len):
        if len_ < (opt.output_w //  opt.stack_num):
            start = len_ * opt.stack_num
            end = (len_+1) * opt.stack_num
            if opt.output_w > opt.output_h:
                temp = torch.zeros(outputs[0]['hm'].shape[0], opt.biconvlstm_input_dim * opt.stack_num\
                                   , opt.output_w, opt.output_h).to(outputs[0]['hm'].device)
                temp[:, :, :opt.output_h, :] = torch.cat([corr_offsetxyz[:, :, start:end, :],
                                                          corr_whxyz[:, :, start:end, :],
                                                          corr_trackingxyz[:, :, start:end, :]],
                                                          dim=2).view([outputs[0]['hm'].shape[0],
                                                                        opt.biconvlstm_input_dim * opt.stack_num,
                                                                        opt.output_h,
                                                                        opt.output_h])
                input_second[:, len_, :, :, :] = temp
            else:
                input_second[:, len_, :, :, :] = torch.cat([corr_offsetxyz[:, :, start:end, :],
                                                            corr_whxyz[:, :, start:end, :],
                                                            corr_trackingxyz[:, :, start:end, :]],
                                                            dim=2).view(single_layer_input_shape)

        elif len_ >= ((opt.output_h + opt.output_w) //  opt.stack_num):
            base = (opt.output_h + opt.output_w) // opt.stack_num
            start = (len_ - base) * opt.stack_num
            end = (len_ - base + 1) * opt.stack_num
            input_second[:, len_, :, :, :] = torch.cat([corr_offsetxyz[:, start:end, :, :],
                                                        corr_whxyz[:, start:end, :, :],
                                                        corr_trackingxyz[:, start:end, :, :]],
                                                       dim=1).view(single_layer_input_shape)

        else:
            base = opt.output_w // opt.stack_num
            start = (len_ - base) * opt.stack_num
            end = (len_ - base + 1) * opt.stack_num
            if opt.output_w < opt.output_h:
                temp = torch.zeros(outputs[0]['hm'].shape[0], opt.biconvlstm_input_dim * opt.stack_num\
                                   , opt.output_w, opt.output_h).to(outputs[0]['hm'].device)
                temp[:, :, :, :opt.output_w] = torch.cat([corr_offsetxyz[:, :, :, start:end],
                                                            corr_whxyz[:, :, :, start:end],
                                                            corr_trackingxyz[:, :, :, start:end]],
                                                           dim=3).view([outputs[0]['hm'].shape[0],
                                                                      opt.biconvlstm_input_dim * opt.stack_num,
                                                                      opt.output_w,
                                                                      opt.output_w])
                input_second[:, len_, :, :, :] = temp
            else:
                input_second[:, len_, :, :, :] = torch.cat([corr_offsetxyz[:, :, :, start:end],
                                                            corr_whxyz[:, :, :, start:end],
                                                            corr_trackingxyz[:, :, :, start:end]],
                                                           dim=3).view(single_layer_input_shape)

    return torch.transpose(input_second, 3, 4)


class Trainer(object):
  def __init__(
          self, opt, model, optimizer, model_second, optimizer_second):
      self.opt = opt
      self.optimizer = optimizer
      self.optimizer_second = optimizer_second
      self.loss_stats, self.loss, self.loss_second = self._get_losses(opt)
      self.loss_stats.append('loss_second')
      self.model_with_loss = ModleWithLoss(opt, model, model_second, self.loss, self.loss_second)

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

  def run_epoch(self, phase, epoch, data_loader):
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
              if k != 'meta':
                  batch[k] = batch[k].to(device=opt.device, non_blocking=True)
          if phase == 'train':
              output, loss, loss_second, loss_stats = model_with_loss(batch)
          else:
              with torch.no_grad():
                  output, loss, loss_second, loss_stats = model_with_loss(batch)
          loss, loss_second = loss.mean(), loss_second.mean()
          if phase == 'train':
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()

              self.optimizer_second.zero_grad()
              loss_second.backward()
              self.optimizer_second.step()

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
              self.debug(batch, output, iter_id, dataset=data_loader.dataset)

          del output, loss, loss_second, loss_stats

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

  def val(self, epoch, data_loader):
      return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
      return self.run_epoch('train', epoch, data_loader)

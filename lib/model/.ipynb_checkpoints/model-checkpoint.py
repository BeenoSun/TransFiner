from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os
import numpy as np

from .networks.dla import DLASeg
from .networks.resdcn import PoseResDCN
from .networks.resnet import PoseResNet
from .networks.dlav0 import DLASegv0
from .networks.generic_network import GenericNetwork
from .BiConvLSTM import BiConvLSTM

_network_factory = {
    'resdcn': PoseResDCN,
    'dla': DLASeg,
    'res': PoseResNet,
    'dlav0': DLASegv0,
    'generic': GenericNetwork
}


def create_model(arch, head, head_conv, opt=None):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    model_class = _network_factory[arch]
    model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)



    model_second = BiConvLSTM((opt.output_h, opt.output_w),
                              torch.tensor(opt.biconvlstm_input_dim * opt.stack_num),
                              opt.biconvlstm_hidden_dim,
                              opt.biconvlstm_kernel_size,
                              torch.tensor(opt.biconvlstm_num_layers),
                              (2*min(opt.output_h,opt.output_w)+max(opt.output_h,opt.output_w))\
                              // opt.stack_num)
    return model, model_second


def load_model(model, model_path, model_second, opt, optimizer=None, optimizer_second=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    state_dict_second = checkpoint['state_dict_second']

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if (state_dict[k].shape != model_state_dict[k].shape) or \
                    (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
                if opt.reuse_hm:
                    print('Reusing parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    if state_dict[k].shape[0] < state_dict[k].shape[0]:
                        model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
                    else:
                        model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
                    state_dict[k] = model_state_dict[k]
                else:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model_second.load_state_dict(state_dict_second, strict=True)
    # resume optimizer parameters
    if optimizer is not None and opt.resume:
        if 'optimizer' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = opt.lr
            for step_ in range(len(opt.lr_step)):
                if start_epoch >= opt.lr_step[step_]:
                    start_lr *= opt.lr_step_weight[step_]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer_second is not None and opt.resume:
        if 'optimizer_second' in checkpoint:
            # optimizer_second.load_state_dict(checkpoint['optimizer_second'])
            start_epoch = checkpoint['epoch']
            start_lr = opt.lr
            for step_ in range(len(opt.lr_step)):
                if start_epoch >= opt.lr_step[step_]:
                    start_lr *= opt.lr_step_weight[step_]
            for param_group in optimizer_second.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer_second with start lr', start_lr)
        else:
            print('No optimizer_second parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, model_second, optimizer_second, start_epoch
    else:
        return model, model_second


def save_model(path, epoch, model, optimizer, model_second, optimizer_second):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    if isinstance(model_second, torch.nn.DataParallel):
        state_dict_second = model_second.module.state_dict()
    else:
        state_dict_second = model_second.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict,
            'state_dict_second': state_dict_second}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    if not (optimizer_second is None):
        data['optimizer_second'] = optimizer_second.state_dict()
    torch.save(data, path)

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
from .fusion_network import REDNet20, REDNet30
import copy
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

    #model_fusion = REDNet30()
    return model#, model_second, model_fusion


def load_model(model, model_path, model_second, opt, optimizer=None, optimizer_second=None,\
               scaler=None, scaler_second=None):
    start_epoch = 0

    # addition
    #model_path = '/home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mix/model_last.pth'
    #opt.crowdma_path = ''

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # /home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/paper_original_custom_dataset_trainhalf/model_70.pth
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    checkpoint_mot17 = torch.load('/home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_fulltrain/mot17_fulltrain.pth')

    state_dict_ = checkpoint_mot17['state_dict'] # checkpoint
    state_dict = {}
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

    # addition
    if opt.crowdma_path != '':
        checkpoint_ano = torch.load(opt.crowdma_path)
        print('loaded {}, epoch {}'.format(opt.crowdma_path, checkpoint_ano['epoch']))
        state_dict_ = checkpoint_ano['state_dict_second']
        model_second.load_state_dict(state_dict_, strict=False)

    #model_second.load_state_dict(state_dict, strict=False)

    # addition for crowdhuman dataset model
    """
    if 'state_dict_second' in checkpoint:
        state_dict_second_ = checkpoint['state_dict_second']
        state_dict_second = {}
        # convert data_parallal to model
        for k in state_dict_second_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict_second[k[7:]] = state_dict_second_[k]
            else:
                state_dict_second[k] = state_dict_second_[k]
        model_second_state_dict = model_second.state_dict()
        # check loaded parameters and created model parameters
        for k in state_dict_second:
            if k in model_second_state_dict:
                if (state_dict_second[k].shape != model_second_state_dict[k].shape) or \
                        (opt.reset_hm and k.startswith('hm') and (state_dict_second[k].shape[0] in [80, 1])):
                    if opt.reuse_hm:
                        print('Reusing parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(
                            k, model_second_state_dict[k].shape, state_dict_second[k].shape))
                        if state_dict_second[k].shape[0] < state_dict_second[k].shape[0]:
                            model_second_state_dict[k][:state_dict_second[k].shape[0]] = state_dict_second[k]
                        else:
                            model_second_state_dict[k] = state_dict_second[k][:model_second_state_dict[k].shape[0]]
                        state_dict_second[k] = model_second_state_dict[k]
                    else:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(
                            k, model_second_state_dict[k].shape, state_dict_second[k].shape))
                        state_dict_second[k] = model_second_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        state_dict_second_final = copy.deepcopy(state_dict_second)
        for k in model_second_state_dict:
          if (k.split('.')[1]).isdigit():
            if int(k.split('.')[1]) < 16:
                state_dict_second_final[k] = state_dict_second[k]
            elif int(k.split('.')[1]) >= 16 and int(k.split('.')[1]) < 30:
                tem = k.split('.')
                tem[1] = str(int(k.split('.')[1]) - 14)
                str_ = ''
                for cha_ in tem:
                    str_ = str_ + cha_ + '.'
                state_dict_second_final[k] = state_dict_second[str_[:-1]]
            elif int(k.split('.')[1]) >= 30 and int(k.split('.')[1]) < 46:
                tem = k.split('.')
                tem[1] = str(int(k.split('.')[1]) - 14)
                str_ = ''
                for cha_ in tem:
                    str_ = str_ + cha_ + '.'
                state_dict_second_final[k] = state_dict_second[str_[:-1]]
            elif int(k.split('.')[1]) >= 46 and int(k.split('.')[1]) < 63:
                tem = k.split('.')
                tem[1] = str(int(k.split('.')[1]) - 15)
                str_ = ''
                for cha_ in tem:
                    str_ = str_ + cha_ + '.'
                state_dict_second_final[k] = state_dict_second[str_[:-1]]
            elif int(k.split('.')[1]) == 63:
                tem = k.split('.')
                tem[1] = str(int(k.split('.')[1]) - 16)
                str_ = ''
                for cha_ in tem:
                    str_ = str_ + cha_ + '.'
                state_dict_second_final[k] = state_dict_second[str_[:-1]]

          if not (k in state_dict_second_final):
            print('No param {}.'.format(k))
            state_dict_second_final[k] = model_second_state_dict[k]
        model_second.load_state_dict(state_dict_second_final, strict=False)
        # addition
    """

    # for mot dataset model
    if opt.crowdma_path == '':
        '''defor_detr = torch.load(
            '/home/beeno/pycharm/py_code/CenterTrack/src_model_transfiner_v1aligner_only_dde/lib/post_transfiner/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth')
        model_second = load_transtrack(model_second, defor_detr)'''
        if 'state_dict_second' in checkpoint:
            print('loading model second！')
            state_dict_second = checkpoint['state_dict_second']
            model_state_dict = model_second.state_dict()
            # check loaded parameters and created model parameters
            for k in state_dict_second:
                if k in model_state_dict:
                    if (state_dict_second[k].shape != model_state_dict[k].shape):
                        print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(k, model_state_dict[
                            k].shape, state_dict_second[k].shape))
                        state_dict_second[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict_second):
                    print('No param {}.'.format(k))
                    state_dict_second[k] = model_state_dict[k]
            model_second.load_state_dict(state_dict_second, strict=False)

            '''# program test only
            tp = state_dict_second
            tf = torch.load(
                '/home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/Transfiner_exp/transpatch_train_realanns_v1/smca_train_decode_sam/model_90.pth')
            for k in tp:
                if 'transpatch' in k:
                    tf['state_dict_second'][k] = tp[k]
            for k in model_state_dict:
                if k in tf['state_dict_second']:
                    model_state_dict[k] = tf['state_dict_second'][k]
                else:
                    print('drop:', k)
            model_second.load_state_dict(model_state_dict, strict=False)'''

            '''if not opt.eva:
                print('loading transtrack!')
                #defor_detr = torch.load('/home/beeno/pycharm/py_code/CenterTrack/src_model_transfiner_v1/lib/post_transfiner/560mot17_crowdhuman.pth')
                defor_detr = torch.load('/home/beeno/pycharm/py_code/CenterTrack/src_model_transfiner_v1aligner_only_dde/lib/post_transfiner/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth')
                model_second = load_transtrack(model_second, defor_detr)'''

        else:
            print('model second not found！')
    '''
    if 'state_dict_fusion' in checkpoint:
        print('loading model fusion！')
        state_dict_fusion = checkpoint['state_dict_fusion']
        model_fusion.load_state_dict(state_dict_fusion, strict=False)
    else:
        print('model fusion not found！')
'''

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
            #start_epoch = checkpoint['epoch']
            #start_lr = opt.lr
            for step_ in range(len(opt.lr_step)):
                if start_epoch >= opt.lr_step[step_]:
                    start_lr *= opt.lr_step_weight[step_]
            for param_group in optimizer_second.param_groups:
                for step_ in range(len(opt.lr_step)):
                    if start_epoch >= opt.lr_step[step_]:
                        param_group['lr'] = param_group['lr'] * opt.lr_step_weight[step_]
            print('Resumed optimizer_second with start lr', start_lr)
        else:
            print('No optimizer_second parameters in checkpoint.')
    '''
    if optimizer_fusion is not None and opt.resume:
        if 'optimizer_fusion' in checkpoint:
            # optimizer_second.load_state_dict(checkpoint['optimizer_second'])
            #start_epoch = checkpoint['epoch']
            #start_lr = opt.lr
            for step_ in range(len(opt.lr_step)):
                if start_epoch >= opt.lr_step[step_]:
                    start_lr *= opt.lr_step_weight[step_]
            for param_group in optimizer_fusion.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer_fusion with start lr', start_lr)
        else:
            print('No optimizer_fusion parameters in checkpoint.')
    '''
    if 'scaler' in checkpoint and opt.resume and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        scaler_second.load_state_dict(checkpoint['scaler_second'])
        #scaler_fusion.load_state_dict(checkpoint['scaler_fusion'])
    if optimizer is not None and scaler is not None:
        return model, optimizer, model_second, optimizer_second, start_epoch,\
               scaler, scaler_second
    else:
        return model, model_second


def save_model(path, epoch, model, optimizer, model_second, optimizer_second, \
               scaler, scaler_second):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    if isinstance(model_second, torch.nn.DataParallel):
        state_dict_second = model_second.module.state_dict()
    else:
        state_dict_second = model_second.state_dict()
    '''
    if isinstance(model_fusion, torch.nn.DataParallel):
        state_dict_fusion = model_fusion.module.state_dict()
    else:
        state_dict_fusion = model_fusion.state_dict()
    '''
    data = {'epoch': epoch,
            'state_dict': state_dict,
            'state_dict_second': state_dict_second}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    if not (optimizer_second is None):
        data['optimizer_second'] = optimizer_second.state_dict()

    if not (scaler is None):
        data['scaler'] = scaler.state_dict()
    if not (scaler_second is None):
        data['scaler_second'] = scaler_second.state_dict()
    torch.save(data, path)

def load_transtrack(model_second, transtrack):

    state_dict = transtrack['model']
    '''state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]'''
    model_state_dict = model_second.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if (state_dict[k].shape != model_state_dict[k].shape):
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            #state_dict[k] = model_state_dict[k]
    model_second.load_state_dict(state_dict, strict=False)

    return model_second


def load_track_model(model, model_path, opt, optimizer=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

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

    return model

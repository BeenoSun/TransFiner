from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
print(torch.distributed.is_available())
print(torch.cuda.is_available())
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model, load_track_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
import numpy as np
from torch.cuda.amp import autocast, GradScaler
#torch.autograd.set_detect_anomaly(True)

def get_optimizer(opt, model): # , model_fusion
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        #optimizer_second = torch.optim.Adam(model_second.parameters(), opt.lr)
        #optimizer_fusion = torch.optim.Adam(model_fusion.parameters(), opt.lr)
    elif opt.optim == 'sgd':
        print('Using SGD')
        optimizer = torch.optim.SGD(
            model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
        #optimizer_second = torch.optim.SGD(
         #   model_second.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
        #optimizer_fusion = torch.optim.SGD(
         #   model_fusion.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
    else:
        assert 0, opt.optim
    return optimizer#, optimizer_second, optimizer_fusion


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    logger = Logger(opt)
    print('Creating model...')
    scaler = GradScaler()
    scaler_second = GradScaler()
    #scaler_fusion = GradScaler()
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    model = load_track_model(model, opt.load_track_model, opt)
    optimizer = get_optimizer(opt, model)
    # '''
    # transfiner implementation by beeno, v1, jan 2022
    from post_transfiner.transfiner import build
    import post_transfiner.utils.misc as utils
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    model_second, criterion_second, postprocessors = build(opt)
    #for param in model_second.parameters():
     #   param.requires_grad = True
    model_without_ddp = model_second
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, opt.lr_backbone_names) and not match_name_keywords(n,
                                                                                                   opt.lr_linear_proj_names) and p.requires_grad],
            "lr": opt.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, opt.lr_backbone_names) and p.requires_grad],
            "lr": opt.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, opt.lr_linear_proj_names) and p.requires_grad],
            "lr": opt.lr * opt.lr_linear_proj_mult,
        }
    ]
    optimizer_second = torch.optim.AdamW(param_dicts, lr=opt.lr,
                                  weight_decay=opt.weight_decay)

    # '''

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, model_second, optimizer_second, \
        start_epoch, scaler, scaler_second = load_model(
            model, opt.load_model, model_second, opt, optimizer, optimizer_second, \
            scaler, scaler_second)
    for param in model.parameters():
        param.requires_grad = False
    #for param in model_second.parameters():
     #   param.requires_grad = False
    trainer = Trainer(opt, model, optimizer, model_second, optimizer_second, criterion_second)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)


    if opt.val_intervals < opt.num_epochs:
        print('Setting up validation data...')
        dataset_val = None #Dataset(opt, 'val')
        sampler_val = None #torch.utils.data.RandomSampler(dataset_val)
        batch_sampler_val = None #torch.utils.data.BatchSampler(sampler_val, opt.batch_size, drop_last=True)
        val_loader = None#torch.utils.data.DataLoader(dataset_val, batch_sampler=batch_sampler_val, num_workers=opt.num_workers, pin_memory=True)

    print('Setting up train data...')
    dataset_train = Dataset(opt, opt.dataty)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, opt.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        num_workers=opt.num_workers, pin_memory=True
    )

    print('Starting training...')

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader, scaler, scaler_second)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer, model_second, optimizer_second,\
                       scaler, scaler_second)
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer, model_second, optimizer_second, \
                       scaler, scaler_second)
            #time.sleep(30)
            #os.system('python test.py tracking --exp_id mot17_half --dataset mot --input_h 672 --input_w 1184 --num_classes 1 --dataset_version 17halfval --gpus 0 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model /home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/model_last.pth --refine_thresh 0.45 --patch_thresh 0 --num_patch_val 1 --eva --nms_thre 0.9 --nms_thre_nd 0.55 --real_num_queries 300 --hm_prob_init 0.1 --dataty_debug val --dec_layers 6 --hungarian --max_age 32 --new_thresh 0.45')
            if epoch > 150:
                with torch.no_grad():
                    log_dict_val, preds = trainer.val(epoch, val_loader)
                    #if opt.eval_val:
                     #   val_loader.dataset.run_eval(preds, opt.save_dir)
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch)
                    logger.write('{} {:8f} | '.format(k, v))
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer, model_second, optimizer_second, \
                       scaler, scaler_second)
        logger.write('\n')
        if epoch in opt.save_point:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer, model_second, optimizer_second, \
                       scaler, scaler_second)
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_second.param_groups:
                param_group['lr'] = param_group['lr'] * (0.1 ** (opt.lr_step.index(epoch) + 1))
    logger.close()


if __name__ == '__main__':

    opt = opts().parse()
    main(opt)

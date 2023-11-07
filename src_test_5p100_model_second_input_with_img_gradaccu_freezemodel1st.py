from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
print(torch.distributed.is_available())
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
import numpy as np
from torch.cuda.amp import autocast, GradScaler


def get_optimizer(opt, model, model_second):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        optimizer_second = torch.optim.Adam(model_second.parameters(), opt.lr)
    elif opt.optim == 'sgd':
        print('Using SGD')
        optimizer = torch.optim.SGD(
            model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
        optimizer_second = torch.optim.SGD(
            model_second.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
    else:
        assert 0, opt.optim
    return optimizer, optimizer_second


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
    model, model_second = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer, optimizer_second = get_optimizer(opt, model, model_second)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, model_second, optimizer_second, start_epoch, scaler, scaler_second = load_model(
            model, opt.load_model, model_second, opt, optimizer, optimizer_second, scaler, scaler_second)

    trainer = Trainer(opt, model, optimizer, model_second, optimizer_second)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)


    # if opt.val_intervals < opt.num_epochs or opt.test:
    #     print('Setting up validation data...')
    #     val_loader = torch.utils.data.DataLoader(
    #         Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
    #         pin_memory=True)
    #
    #     if opt.test:
    #         _, preds = trainer.val(0, val_loader)
    #         val_loader.dataset.run_eval(preds, opt.save_dir)
    #         return

    print('Setting up train data...')
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )

    print('Starting training...')

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        if epoch == 1:
            for param in model.parameters():
                param.requires_grad = False
            for param in model_second.hm_corr_network.parameters():
                param.requires_grad = False
        elif epoch == 3:
            for param in model.parameters():
                param.requires_grad = True
            for param in model_second.hm_corr_network.parameters():
                param.requires_grad = True
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader, scaler, scaler_second)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer, model_second, optimizer_second, scaler, scaler_second)
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer, model_second, optimizer_second, scaler, scaler_second)
            os.system('python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --gpus 0,1 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --resume --load_model /home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/model_last.pth')
            # with torch.no_grad():
            #     log_dict_val, preds = trainer.val(epoch, val_loader)
            #     if opt.eval_val:
            #         val_loader.dataset.run_eval(preds, opt.save_dir)
            # for k, v in log_dict_val.items():
            #     logger.scalar_summary('val_{}'.format(k), v, epoch)
            #     logger.write('{} {:8f} | '.format(k, v))
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer, model_second, optimizer_second, scaler, scaler_second)
        logger.write('\n')
        if epoch in opt.save_point:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer, model_second, optimizer_second, scaler, scaler_second)
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_second.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

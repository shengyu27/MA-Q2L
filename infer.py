import argparse
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import skimage
import torchvision.utils

import random
import datetime
import time
from typing import List
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import _init_paths
from lib.dataset.get_dataset import get_datasets


from lib.utils.logger import setup_logger
import lib.models
import lib.models.aslloss
from lib.models.MAQ2L import build_MAQ2L
from lib.utils.metrics_sewerML import return_mAP
from lib.utils.misc import clean_state_dict
from lib.utils.slconfig import get_raw_dict


def parser_args():
    available_models = ['MAQ2L-R101-448', 'MAQ2L-R101-576', 'MAQ2L-TResL-448', 'MAQ2L-TResL_22k-448', 'MAQ2L-SwinL-384', 'MAQ2L-CvT_w24-384']

    parser = argparse.ArgumentParser(description='MAQ2L for multilabel classification')
    parser.add_argument('--dataname', help='dataname', default='sewerml', choices=['sewerml'])
    parser.add_argument('--dataset_size', help='datset_size', default=0.0625, choices=['1', '0.0625'])
    parser.add_argument('--dataset_dir', help='dir of dataset',
                        default='./dataset')
    
    parser.add_argument('--img_size', default=448, type=int,
                        help='image size. default(448)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='MAQ2L-R101-448',
                        choices=available_models,
                        help='model architecture: ' +
                            ' | '.join(available_models) +
                            ' (default: MAQ2L-R101-448)')
    parser.add_argument('--config', type=str, help='config file',
                        )
    parser.add_argument('--checkpoint', metavar='DIR',
                        default='./checkpoint/model_best.pth.tar',
                        help='path to output folder')
    parser.add_argument('--output', metavar='DIR',
                        default='./output',
                        help='path to output folder')

    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--loss', metavar='LOSS', default='asl', 
                        choices=['asl'],help='loss functin')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')

    parser.add_argument('--num_class', default=17, type=int,
                        help="Number of classes.")
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')

    parser.add_argument('--weight_top', default=4, type=float,
                        help='top k for loss weight')
    parser.add_argument('--dy_weight', default=False, type=bool,
                        help='use dynamic weight top ')
    parser.add_argument('--eps', default=1e-5, type=float,
                    help='eps for focal loss (default: 1e-5)')

    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help='use mixture precision.')
    # data aug
    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using oridinary norm of [0,0,0] and [1,1,1] for mean and std.')


    # * Transformer

    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
    args = parser.parse_args()

    # update parameters with pre-defined config file
    if args.config:
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k,v in cfg_dict.items():
            setattr(args, k, v)

    return args

def get_args():
    args = parser_args()
    return args


best_mAP = 0

def main():
    args = get_args()
    
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    
    # set output dir and logger
    if not args.output:
        args.output = (f"logs/{args.arch}-{datetime.datetime.now()}").replace(' ', '-')
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: "+' '.join(sys.argv))


    # save config to outputdir
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    return main_worker(args, logger)

def main_worker(args, logger):
    global best_mAP

    # build model
    model = build_MAQ2L(args)
    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    criterion = lib.models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=True,
        eps=args.eps,
    )

    weights_path = args.checkpoint
    checkpoint = torch.load(weights_path, map_location=torch.device(dist.get_rank()))#['state_dict']
    state_dict = clean_state_dict(checkpoint['state_dict'])

    model.module.load_state_dict(state_dict, strict = True)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))
    #         state_dict = clean_state_dict(checkpoint['state_dict'])
    #         model.module.load_state_dict(state_dict, strict=True)
    #         del checkpoint
    #         del state_dict
    #         torch.cuda.empty_cache()
    #     else:
    #         logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    _, val_dataset = get_datasets(args)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    # for eval only
    validate(val_loader, model, criterion, args, logger)

    return


@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    saved_data = []
    with torch.no_grad():
        start = time.time()
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)


                loss = criterion(output, target)
                output_sm = torch.sigmoid(output)
                #write_pre(i, output_sm.detach().cpu().numpy(), str_file)
            # record loss
            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # save some data
            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            saved_data.append(_item)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)
        end_time = time.time()

        # print('time:', (end_time-start)/8127)
        #logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()

        # save results
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        logger.info('inference have done!')




##################################################################################

def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpget_datasetsoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def kill_process(filename:str, holdpid:int) -> List[str]:
    # used for training only.
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

if __name__ == '__main__':
    main()

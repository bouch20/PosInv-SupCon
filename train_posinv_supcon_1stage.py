
import argparse

import random
import shutil
import time
import warnings
import os
import numpy as np
import math

import torch
import torch.backends.cudnn as cudnn

from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets

from PIL import Image

from utils.utils import ProgressMeter, AverageMeter, adjust_learning_rate, warmup_learning_rate, BalancedBatchSampler
from networks.resnet import SupConResNetWithEmbedding
from losses import PosInvSupConLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,150,175',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cub-200-2011',
                        choices=['cifar10', 'cub-200-2011', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon'], help='choose method')
                        
    # proxy num
    parser.add_argument('--proxy_num', type=int, default=1,
                        help='proxy number per class')
    parser.add_argument('--over_proxy_num', type=int, default=0,
                        help='proxy number per class')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # feature dimention
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='feature dimention')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--validation', action='store_true',
                        help='using validation')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='split training set for validation set')
    parser.add_argument('--emb_init', action='store_true',
                        help='Embedding Layer initialize')

    parser.add_argument('--balanced_sampler', action='store_true',
                        help='Embedding Layer initialize')
    parser.add_argument('--sampler_class_num', type=int, default=4,
                        help='Embedding Layer initialize')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './models/{}'.format(opt.dataset)
    opt.tb_path = './tensorboards/{}'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_featdim_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.feat_dim)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.proxy_num > 1:
        opt.model_name = '{}_proxy_num_{}'.format(opt.model_name, opt.proxy_num)

    if opt.over_proxy_num > 0:
        opt.model_name = '{}_over_proxy_num_{}'.format(opt.model_name, opt.over_proxy_num)

    if opt.emb_init:
        opt.model_name = '{}_embinit'.format(opt.model_name)
    if opt.balanced_sampler:
        if not (opt.batch_size % opt.sampler_class_num == 0):
            print("batch_size が割り切れませんでした")
            exit(1)
        print("sampler 実行")
        opt.model_name = '{}_sampler'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_model(opt):
    print(f"model : {opt.model}")
    model = SupConResNetWithEmbedding(name=opt.model, feat_dim=opt.feat_dim, num_classes=(200*opt.proxy_num) + opt.over_proxy_num)
    criterion = PosInvSupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(args.seed)


def get_dataset_loaders(args, traindir="../data/CUB_200_2011/image_folder/train", valdir="../data/CUB_200_2011/image_folder/test"):
    
    train_transforms = transforms.Compose([
                    transforms.Resize((600, 600), Image.BILINEAR),
                    transforms.RandomCrop((448, 448)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    trainval_dataset = datasets.ImageFolder(
        traindir,
        train_transforms
        )

    train_sampler = None

    if args.validation:
        targets = np.array(trainval_dataset.targets)
        train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), test_size=args.val_size, stratify=targets, random_state=42)

        print(len(train_indices))
        train_dataset = torch.utils.data.Subset(trainval_dataset, train_indices)
        val_dataset   = torch.utils.data.Subset(trainval_dataset, test_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, sampler=None)

        return train_loader, val_loader
    else:
        
        if args.balanced_sampler:
            # train_sampler = BalancedBatchSampler(trainval_dataset, args.sampler_class_num, int(args.batch_size // args.sampler_class_num))
            train_sampler = BalancedBatchSampler(trainval_dataset, 8, 4)
            train_loader = torch.utils.data.DataLoader(
                trainval_dataset, num_workers=args.num_workers, pin_memory=True, batch_sampler=train_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(
                trainval_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        return train_loader

def train(train_loader, model, criterion, optimizer, epoch, args, writer, n_iter, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for idx, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        bsz = target.shape[0]
        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        emb_idx = torch.arange(0, 200*args.proxy_num, dtype=torch.int64).cuda()
        emb_label = torch.fmod(emb_idx, 200).cuda()
        if args.over_proxy_num > 0:
            emb_idx = torch.cat([emb_idx, torch.arange(200*args.proxy_num, (200*args.proxy_num)+args.over_proxy_num).cuda()])
            emb_label = torch.cat([emb_label, torch.add(torch.ones(args.over_proxy_num), 200).cuda()])
        features, emb_feature = model(images, emb_idx)
        features2 = torch.cat([features.unsqueeze(1), emb_feature.unsqueeze(1)], dim=0)
        labels2 = torch.cat([target, emb_idx], dim=0)
        loss = criterion(features2, labels=labels2)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx+1) % args.print_freq == 0:
            progress.display(idx+1)
    
    writer.add_scalar('train/supcon_loss', losses.avg, n_iter)
    return losses.avg


def validate(val_loader, model, criterion, optimizer, epoch, args, writer, n_iter, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to val mode
    model.eval()

    end = time.time()
    for idx, (images, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        bsz = target.shape[0]

        with torch.no_grad():
            # compute loss
            emb_idx = torch.arange(0, 200, dtype=torch.int64).cuda()
            features, emb_feature = model(images, emb_idx)
            features2 = torch.cat([features.unsqueeze(1), emb_feature.unsqueeze(1)], dim=0)
            labels2 = torch.cat([target, emb_idx], dim=0)
            loss = criterion(features2, labels=labels2)

        # update metric
        losses.update(loss.item(), bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx+1) % args.print_freq == 0:
            progress.display(idx+1)
    
    writer.add_scalar('val/supcon_loss', losses.avg, n_iter)
    return losses.avg


def save_checkpoint(state, is_best, fileprefix='model'):
    torch.save(state, f"weights/{fileprefix}_checkpoint.pth.tar")
    if is_best:
        shutil.copyfile(f"weights/{fileprefix}_checkpoint.pth.tar", f"weights/{fileprefix}_model_best.pth.tar")

def model_emb_init(model, num_class, feat_dim):

    label_range = torch.arange(0, num_class, dtype=torch.int64).contiguous().view(-1, 1)
    emb_range = torch.arange(0, feat_dim, dtype=torch.int64).contiguous().view(-1, 1)
    init_emb_value = torch.eq(label_range, emb_range.T).float().cuda()

    model.embedding.weight.data.copy_(init_emb_value)
    return model

def main():
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    print(f"epochs={args.epochs}")
    num_class = 200
    model, criterion = set_model(args)
    if args.emb_init:
        if num_class < args.feat_dim:
            print("emb initialize done")
            model = model_emb_init(model, num_class, args.feat_dim)

    writer = SummaryWriter(log_dir=args.tb_folder, flush_secs=2)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # scaler = torch.cuda.amp.GradScaler()
    scaler = None
    if args.validation:
        train_loader, val_loader = get_dataset_loaders(args)
    else:
        train_loader = get_dataset_loaders(args)
    best_loss = 80.0

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        loss1 = train(train_loader, model, criterion, optimizer, epoch, args, writer, epoch, scaler)
        if args.validation:
            loss1 = validate(val_loader, model, criterion, optimizer, epoch, args, writer, epoch, scaler)

        # remember best acc@1 and save checkpoint
        is_best = best_loss > loss1
        best_loss = min(loss1, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, fileprefix=args.model_name)
    print(f"save best model path = weights/{args.model_name}_model_best.pth.tar")
    writer.close()

if __name__ == "__main__":
    main()

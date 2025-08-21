
import argparse

import random
import shutil
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets

from PIL import Image

from utils.utils import ProgressMeter, AverageMeter, adjust_learning_rate, warmup_learning_rate, accuracy
from networks.resnet import SupConResNet, LinearClassifier, SupConResNetWithEmbedding

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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # proxy num
    parser.add_argument('--proxy_num', type=int, default=1,
                        help='proxy number per class')
    parser.add_argument('--over_proxy_num', type=int, default=0,
                        help='proxy number per class')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cub-200-2011',
                        choices=['cifar10', 'cub-200-2011'], help='dataset')

    # feature dimention
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='feature dimention')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    parser.add_argument('--validation', action='store_true',
                        help='using validation')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='split training set for validation set')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'Linear_{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'cub-200-2011':
        opt.n_cls = 200
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_model(opt):
    model = SupConResNetWithEmbedding(name=opt.model, feat_dim=opt.feat_dim, num_classes=(200*opt.proxy_num) + opt.over_proxy_num)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['state_dict']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(args.seed)

def get_dataset_loaders(args, traindir="../data/CUB_200_2011/image_folder/train", valdir="../data/CUB_200_2011/image_folder/test"):

    trainval_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    train_sampler = None
    if args.validation:
        targets = np.array(trainval_dataset.targets)
        train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), test_size=0.2, stratify=targets, random_state=42)

        print(len(train_indices))
        train_dataset = torch.utils.data.Subset(trainval_dataset, train_indices)
        val_dataset   = torch.utils.data.Subset(trainval_dataset, test_indices)

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, sampler=None)

        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        return train_loader, test_loader

def train(train_loader, model, classifier, criterion,
          optimizer, epoch, args, writer):
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
    model.eval()
    classifier.train()

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
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx+1) % args.print_freq == 0:
            progress.display(idx+1)
    
    # writer.add_scalar('train/loss_ce', losses.avg, epoch)
    # writer.add_scalar('train/Accuracy', top1.avg, epoch)
    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, args, writer, n_iter):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = classifier(model.encoder(images))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % args.print_freq == 0:
                progress.display(idx+1)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
        # writer.add_scalar('validation/Loss', losses.avg, n_iter)
        # writer.add_scalar('validation/Accuracy', top1.avg, n_iter)

    return losses.avg, top1.avg

def save_checkpoint(state, is_best, fileprefix='model'):
    torch.save(state, f"weights/{fileprefix}_checkpoint.pth.tar")
    if is_best:
        shutil.copyfile(f"weights/{fileprefix}_checkpoint.pth.tar", f"weights/{fileprefix}_model_best.pth.tar")

def main():
    args = parse_option()

    set_seed(args)
    
    print(f"epochs={args.epochs}")
    ngpus_per_node = torch.cuda.device_count()

    model, classifier, criterion = set_model(args)

    model.eval()
    classifier.train()
    # writer = SummaryWriter(log_dir=args.tb_folder, flush_secs=2)
    writer = None
    # Data loading code
    # define loss function (criterion) and optimizer

    optimizer = torch.optim.SGD(classifier.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.validation:
        train_loader, val_loader, test_loader = get_dataset_loaders(args)
    else:
        train_loader, test_loader = get_dataset_loaders(args)
    best_acc = 0
    test_acc = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, args, writer)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        # eval for one epoch
        if args.validation:
            _loss, acc1 = validate(val_loader, model, classifier, criterion, args, writer, epoch)
        else:
            _loss, acc1 = validate(test_loader, model, classifier, criterion, args, writer, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)
        if is_best:
            if args.validation:
                _loss, test_acc = validate(test_loader, model, classifier, criterion, args, writer, epoch)
                print('test_acc@1: {:.2f}'.format(test_acc))
            else:
                print('best accuracy: {:.2f}'.format(best_acc))            

    print('best accuracy: {:.2f}'.format(best_acc))

if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np
import shutil
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from dataset.oxford_dataset import oxford_dataset
from scripts.configure_logging import configure_logging, get_learning_rate
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from PIL import Image
import pandas as pd
from pathlib import Path
from scripts.testing_functions import calc_accuracy
import pretrainedmodels


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=8, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=8, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_reduce', default=0.3, type=float, help='lr reduce on plateau factor')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--visdom_server', type=str, default='dccxc112.pok.ibm.com', help='visdom server address')
parser.add_argument('--env_name', type=str, default='oxfordNasnetAll', help='Environment name for naming purposes')
# Checkpoints

parser.add_argument('-c', '--checkpoint', default='C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/oxford/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/inception_trainDBIInceptionHalf2019;3;24;3;7best.pkl
# C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/DBINasnet192019;3;24;16;10best.pkl
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnet',
                    help='model architecture:(resnet18/ inception/ nasnet)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--images_path', default='./data/oxford_pets/images', type=str, metavar='PATH', help='path to train ccsn data')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--debug_mode', type=int, default=0, help='activate fast debug mode?(default: 0) 0(no)/1(yes)')

parser.add_argument('--freezeLayers', type=str, default='half',
                    help='how many layers should we freeze? options:none/half/all')
parser.add_argument('--freezeLayersNum', type=int, default=19,
                    help='how many layers should we freeze? options:int')

parser.add_argument('--evaluate', type=int, default=0, help='evaluate (ONLY!) model on validation and test sets? 0(no)/1(yes)')
parser.add_argument('--random_angle', type=float, default=10, help='Angle of random augmentation.')
parser.add_argument('--pretrained', type=int, default=1, help='use pre-trained model 0(no)/1(yes)')

#Device options

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)
# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    # args.manualSeed = random.randint(1, 10000)
    args.manualSeed = 5
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_val_acc = 0  # best test accuracy

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def main():
    global best_val_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Data loading code
    if args.arch == "inception":
        scaler = 350
        img_size = 299
        norm_trans = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    elif args.arch == "nasnet":
        scaler = 354
        img_size = 331
        norm_trans = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    else:
        scaler = 280
        img_size = 224
        norm_trans = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    # train_transform = transforms.Compose(
    #     [
    #         # transforms.Resize((img_size, img_size)),
    #         transforms.Resize(scaler),
    #         transforms.RandomResizedCrop(img_size),
    #         transforms.RandomRotation(degrees=args.random_angle, resample=Image.BILINEAR),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor()
    #     ]
    # )
    train_transform = transforms.Compose(
        [
            # transforms.Resize((img_size, img_size)),
            transforms.Resize(scaler),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomRotation(degrees=args.random_angle, resample=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm_trans
        ]
    )
    test_transform = transforms.Compose(
        [
            # transforms.Resize((img_size, img_size)),
            transforms.Resize(scaler),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm_trans
        ]
    )

    # create train and test datasets
    imgs_path = Path(args.images_path)
    trainODDataset = oxford_dataset(imgs_path, transform=train_transform)
    validation_split = 0.2
    dataset_size = len(trainODDataset)
    if args.debug_mode == 1:
        indices = list(range(int(dataset_size/18)))
    else:
        indices = list(range(dataset_size))
    split = int(np.floor(validation_split * len(indices)))
    np.random.seed(args.manualSeed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch,
                                             sampler=valid_sampler)

    out_size = 25
    print("=> using pre-trained model '{}'".format(args.arch))
    if args.arch == 'inception':
        # model = models.inception_v3(pretrained=True, aux_logits=False)
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_size)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, out_size)
    elif args.arch == 'resnet18':
        # model = models.resnet18(pretrained=True)
        model = models.resnet101(pretrained=True)
        # model = models.alexnet(pretrained=True)
        # model = models.vgg16(pretrained=True)
        # switch to True if using VGG/ alexnet
        if False:
            model.classifier[6] = nn.Linear(4096, out_size)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, out_size)
    elif args.arch == 'nasnet':
        # model = models.inception_v3(pretrained=True, aux_logits=False)
        model = pretrainedmodels.nasnetalarge(num_classes=1000, pretrained='imagenet')
        if args.freezeLayers == 'all':
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, out_size)
    else:
        raise NotImplementedError("only inception_v3 and resnet18 are available at the moment")
    criterion = nn.CrossEntropyLoss()


    if use_cuda:
        print("using gpu")
        model.cuda()
        criterion.cuda()


    # cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True, factor=args.lr_reduce)
    env_name = args.env_name
    now = datetime.datetime.now()
    vis_file_out = env_name + str(now.year) + ';' + str(now.month) + ';' + \
                   str(now.day) + ';' + str(now.hour) + ';' + str(now.minute)
    # TODO: save to logs to disk
    # notes_logger = visdomLogger.VisdomTextLogger(update_type='APPEND', server=args.visdom_server, env=env_name)
    # notes_logger.log("Start time: {}".format(now))
    # notes_logger.log("Args: {}".format(args))
    # notes_logger.log("data lenght: {}".format(len(trainODDataset.trainImgPathLabels)))



    log_filename = './logs/' + str(vis_file_out) + '.txt'
    logger = configure_logging(log_filename)
    logger.info(args)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    # print network arch to decide how many layers to freeze
    for id, child in enumerate(model.children()):
        for name, param in child.named_parameters():
            print(id, ': ', name)
    exit()
    # freeze layers
    if args.freezeLayers == 'none':
        print("not freezing anything")
        logger.info("not freezing anything")
    elif args.freezeLayers == 'half':
        print("freezing first layers only")
        logger.info("freezing first layers only")
        if args.arch == 'inception':
            special_child = 5
        elif args.arch == 'resnet18':
            special_child = 5
        elif args.arch == 'nasnet':
            special_child = args.freezeLayersNum
            # special_child = 19
        child_counter = 0
        for child in model.children():
            if child_counter < special_child:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                break
            child_counter += 1
    elif args.freezeLayers == 'all':
        print("freezing all layers except for the top layer")
        logger.info("freezing all layers except for the top layer")
        if args.arch == 'inception':
            last_child = 17
        elif args.arch == 'resnet18':
            last_child = 9
        elif args.arch == 'nasnet':
            last_child = 0
        child_counter = 0
        for child in model.children():
            if child_counter < last_child:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                break
            child_counter += 1
    else:
        raise NotImplementedError("only none/ half/ all options are available at the moment")

    if args.evaluate == 1:
        print('\nEvaluation only')
        logger.info('\nEvaluation only')

        val_loss, val_acc = val(val_loader, model, criterion, use_cuda, scheduler)
        print("Val Loss: {}, Val Acc: {}".format(val_loss, val_acc))
        logger.info("Val Loss: {}, Val Acc: {}".format(val_loss, val_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # notes_logger.log("Epoch: {} of {}, LR: {}".format(epoch+1, args.epochs, state['lr']))
        logger.info("Epoch: {} of {}, LR: {}".format(epoch+1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        print("finished train, starting test")
        logger.info("finished train, starting test")
        val_loss, val_acc = val(val_loader, model, criterion, use_cuda, scheduler)
        # log loss and accuracy
        print("learning rate: {}".format(get_learning_rate(optimizer)))
        logger.info("learning rate: {}".format(get_learning_rate(optimizer)))
        print("train loss: {}, train accuracy: {}".format(train_loss, train_acc))
        logger.info("train loss: {}, train accuracy: {}".format(train_loss, train_acc))
        print("val loss: {}, val accuracy: {}".format(val_loss, val_acc))
        logger.info("val loss: {}, val accuracy: {}".format(val_loss, val_acc))

        # save model
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        filename = str(vis_file_out)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'acc': val_acc, 'best_val_acc': best_val_acc,
                         'optimizer': optimizer.state_dict()
                         }, is_best, epoch, checkpoint=args.checkpoint, filename=filename)

    print('Best val acc: '.format(best_val_acc))
    logger.info('Best val acc: '.format(best_val_acc))


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    total_loss = 0
    total_acc = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batches_done = epoch * len(train_loader) + batch_idx
        optimizer.zero_grad()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # targets = targets.type(FloatTensor)
        # targets = targets.type(DoubleTensor)
        # compute output
        if args.arch == "inception":
            outputs, aux_logits = model(inputs)
            # outputs = outputs.type(DoubleTensor)
            loss = criterion(outputs, targets)
            # loss2 = criterion(aux_logits, targets)
            # loss = (loss1 + loss2) / 2
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        # measure accuracy and record loss
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        outputs_soft = F.softmax(outputs, dim=1)
        outputs_clone = outputs_soft.clone().data.cpu().numpy()
        targets_clone = targets.clone().data.cpu().numpy()

        total_acc += [calc_accuracy(outputs_clone, targets_clone)]

    #     if batch_idx == 0:
    #         outputs_soft = F.softmax(outputs.clone(), dim=1)
    #         outputs_clone = outputs_soft.data.cpu().numpy()
    #         targets_clone = targets.clone().data.cpu().numpy()
    #         all_batch_outputs = outputs_clone
    #         all_batch_targets = targets_clone
    #         # all_batch_targets = targets.clone().data.cpu().numpy()
    #     else:
    #         outputs_soft = F.softmax(outputs.clone(), dim=1)
    #         outputs_clone = outputs_soft.data.cpu().numpy()
    #         targets_clone = targets.clone().data.cpu().numpy()
    #         all_batch_outputs = np.concatenate((all_batch_outputs, outputs_clone), axis=0)
    #         all_batch_targets = np.concatenate((all_batch_targets, targets_clone), axis=0)
    # accuracy = calc_accuracy(all_batch_outputs, all_batch_targets)
    accuracy = sum(total_acc) / len(total_acc)
    return total_loss, accuracy


def val(val_loader, model, criterion, cuda, scheduler):
    # switch to evaluate mode
    model.eval()
    total_loss = 0
    total_acc = []
    # print("loader length: ", len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # compute output
        if args.arch == "inception":
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        total_loss += loss.item()

        outputs_soft = F.softmax(outputs.clone(), dim=1)
        outputs_clone = outputs_soft.data.cpu().numpy()
        targets_clone = targets.clone().data.cpu().numpy()
        total_acc += [calc_accuracy(outputs_clone, targets_clone)]

    #     if batch_idx == 0:
    #         outputs_soft = F.softmax(outputs.clone(), dim=1)
    #         outputs_clone = outputs_soft.data.cpu().numpy()
    #         targets_clone = targets.clone().data.cpu().numpy()
    #         all_batch_outputs = outputs_clone
    #         all_batch_targets = targets_clone
    #     else:
    #         outputs_soft = F.softmax(outputs.clone(), dim=1)
    #         outputs_clone = outputs_soft.data.cpu().numpy()
    #         targets_clone = targets.clone().data.cpu().numpy()
    #         all_batch_outputs = np.concatenate((all_batch_outputs, outputs_clone), axis=0)
    #         all_batch_targets = np.concatenate((all_batch_targets, targets_clone), axis=0)
    # accuracy = calc_accuracy(all_batch_outputs, all_batch_targets)
    accuracy = sum(total_acc) / len(total_acc)
    scheduler.step(total_loss / len(val_loader))
    return total_loss, accuracy


def save_checkpoint(state, is_best, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    file_path = checkpoint + filename + 'last' + '.pkl'
    with open(file_path, 'wb') as f:
        torch.save(state, f)
    if is_best:
        best_file_path = checkpoint + filename + 'best' + '.pkl'
        with open(best_file_path, 'wb') as f:
            torch.save(state, f)


if __name__ == '__main__':
    main()

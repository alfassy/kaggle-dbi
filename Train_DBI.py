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
from torch.utils.data import Dataset,DataLoader,TensorDataset,ConcatDataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from dataset.kaggle_dataset import DBI_dataset, DBI_dataset_ensemble
from scripts.configure_logging import configure_logging, get_learning_rate
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from PIL import Image
import pandas as pd
from pathlib import Path
from scripts.testing_functions import calc_accuracy
import pretrainedmodels
from scripts.getting_features import get_feature_vecs, DropClassifier

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
parser.add_argument('--train_batch', default=4, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=4, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_reduce', default=0.3, type=float, help='lr reduce on plateau factor')
parser.add_argument('--drop', '--dropout', default=0.5, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--visdom_server', type=str, default='dccxc112.pok.ibm.com', help='visdom server address')
parser.add_argument('--env_name', type=str, default='DBINasnet19OxfordPretrained', help='Environment name for naming purposes')
# Checkpoints

parser.add_argument('-c', '--checkpoint', default='C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# kaggle_dbiDBINasnet192019;3;24;18;29best.pkl
# C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/inception_trainDBIInceptionHalf2019;3;24;3;7best.pkl
# C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/DBINasnet192019;3;24;16;10best.pkl
# Architecture
parser.add_argument('--transfer_oxford', default='',
                    type=str, metavar='PATH', help='path to model pretrained on oxford pets checkpoint (default: none)')
parser.add_argument('--pretrained_kaggle', default='C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/DBINasnet19OxfordPretrained2019;3;27;0;4best.pkl',
                    type=str, metavar='PATH', help='path to model pretrained kaggle for the last layer (default: none)')
# C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/oxford/oxfordNasnet192019;3;24;19;21best.pkl
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnet',
                    help='model architecture:(resnet18/ inception/ nasnet)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--train_path', default='./data/kaggle_dbi/train', type=str, metavar='PATH', help='path to train ccsn data')
parser.add_argument('--train_labels_path', default='./data/kaggle_dbi/labels.csv', type=str, metavar='PATH', help='path to train ccsn data')
parser.add_argument('--test_path', default='./data/kaggle_dbi/test', type=str, metavar='PATH', help='path to val ccsn data')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--debug_mode', type=int, default=0, help='activate fast debug mode?(default: 0) 0(no)/1(yes)')

parser.add_argument('--freezeLayers', type=str, default='all',
                    help='how many layers should we freeze? options:none/half/all')
parser.add_argument('--freezeLayersNum', type=int, default=19,
                    help='how many layers should we freeze? options:int')

parser.add_argument('--evaluate', type=int, default=0, help='evaluate (ONLY!) model on validation and test sets? 0(no)/1(yes)')
parser.add_argument('--use_saved_feature_vecs', type=int, default=1, help='Save feature vecs for entire data first for better performance in long runs 0(no)/1(yes)')
parser.add_argument('--random_angle', type=float, default=10, help='Angle of random augmentation.')
parser.add_argument('--pretrained', type=int, default=1, help='use pre-trained model 0(no)/1(yes)')
parser.add_argument('--ensemble', type=int, default=0, help='should we use model ensemble? 0(no)/1(yes)')
parser.add_argument('--ensemble_path_a', default='/dccstor/alfassy/saved_models/inception_trainIncHalf001NoAux2019.3.7.11:51best', type=str, metavar='PATH', help='path to 1st ensemble model (default: none)')
parser.add_argument('--ensemble_path_b', default='/dccstor/alfassy/saved_models/resnet18_trainRes101Half_0012019.3.6.8:28best', type=str, metavar='PATH', help='path to 2nd ensemble model (default: none)')

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
    env_name = args.env_name
    now = datetime.datetime.now()
    vis_file_out = env_name + str(now.year) + ';' + str(now.month) + ';' + \
                   str(now.day) + ';' + str(now.hour) + ';' + str(now.minute)
    log_filename = './logs/' + str(vis_file_out) + '.txt'
    logger = configure_logging(log_filename)
    logger.info(args)

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
    train_transform2 = transforms.Compose(
        [
            # transforms.Resize((img_size, img_size)),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
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
    fp_train = Path(args.train_path)
    fp_test = Path(args.test_path)
    df_train = pd.read_csv(args.train_labels_path, index_col=0)
    df_test = pd.read_csv('./data/kaggle_dbi/sample_submission.csv', index_col=0)

    if args.ensemble == 1:
        trainODDataset = DBI_dataset_ensemble(fp_train, df_train, df_test, is_train=True, transform=train_transform2)
        testODDataset = DBI_dataset_ensemble(fp_test, df_train, df_test, is_train=False, transform=test_transform)
    else:
        trainODDataset = DBI_dataset(fp_train, df_train, df_test, is_train=True, transform=train_transform2)
        testODDataset = DBI_dataset(fp_test, df_train, df_test, is_train=False, transform=test_transform)

    # validation_split = 0.2
    # dataset_size = len(trainODDataset)
    # if args.debug_mode == 1:
    #     indices = list(range(int(dataset_size/18)))
    # else:
    #     indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * len(indices)))
    # np.random.seed(args.manualSeed)
    # np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)

    train_loader_all = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch, shuffle=False)
    # train_loader = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch,
    #                                            sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch,
    #                                          sampler=valid_sampler)
    # test_dataset_size = len(testODDataset)
    # test_indices = list(range(int(test_dataset_size / 18)))
    test_indices = list(range(args.test_batch))
    test_sampler = SubsetRandomSampler(test_indices)
    if args.debug_mode == 1:
        test_loader = torch.utils.data.DataLoader(testODDataset, batch_size=args.test_batch, sampler=test_sampler,
                                                  shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(testODDataset, batch_size=args.test_batch, shuffle=False)

    out_size = 120
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
        if args.transfer_oxford:
            model.last_linear = nn.Linear(num_ftrs, 25)
        else:
            model.last_linear = nn.Linear(num_ftrs, out_size)
    else:
        raise NotImplementedError("only inception_v3 and resnet18 are available at the moment")
    criterion = nn.CrossEntropyLoss()

    # load pretrained oxford pets model
    if args.transfer_oxford:
        # Load checkpoint.
        print('==> Resuming from pretrained oxford pets checkpoint..')
        logger.info('==> Resuming from pretrained oxford pets checkpoint..')
        assert os.path.isfile(args.transfer_oxford), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.transfer_oxford)
        model.load_state_dict(checkpoint['state_dict'])
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, out_size)

    if use_cuda:
        print("using gpu")
        model.cuda()
        criterion.cuda()


    # cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
    optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True, factor=args.lr_reduce)


    if args.ensemble == 1:
        model_a = models.inception_v3(pretrained=True)
        num_ftrs = model_a.fc.in_features
        model_a.fc = nn.Linear(num_ftrs, out_size)
        num_ftrs = model_a.AuxLogits.fc.in_features
        model_a.AuxLogits.fc = nn.Linear(num_ftrs, out_size)
        model_b = models.resnet101(pretrained=True)
        num_ftrs = model_b.fc.in_features
        model_b.fc = nn.Linear(num_ftrs, out_size)
        print('==> doing ensemble..')
        logger.info('==> doing ensemble..')
        assert os.path.isfile(args.ensemble_path_a), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.ensemble_path_a)
        checkpointa = torch.load(args.ensemble_path_a)
        checkpointb = torch.load(args.ensemble_path_b)
        # best_val_acc = checkpoint['best_val_acc']
        # start_epoch = checkpoint['epoch']
        model_a.load_state_dict(checkpointa['state_dict'])
        # optimizera.load_state_dict(checkpointa['optimizer'])
        model_b.load_state_dict(checkpointb['state_dict'])
        # optimizerb.load_state_dict(checkpointb['optimizer'])
        model_a.cuda()
        model_b.cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    # print network arch to decide how many layers to freeze
    # for id, child in enumerate(model.children()):
    #     for name, param in child.named_parameters():
    #         print(id, ': ', name)
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
        last_child = 0
        if args.arch == 'inception':
            last_child = 17
        elif args.arch == 'resnet18':
            last_child = 9
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

    if args.use_saved_feature_vecs == 1:
        args.epochs = 10000
        # load feature vectors into memory
        # feature_vecs_inputs, feature_vecs_targets = get_feature_vecs(train_loader_all, model)
        file_path_inputs = 'C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/feature_vecs_inputs.pkl'
        file_path_inputs_npy = 'C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/feature_vecs_inputs.npy'
        # with open(file_path_inputs, 'wb') as f:
        #     torch.save(feature_vecs_inputs, f)
        # np.save(file_path_inputs_npi, feature_vecs_inputs)
        feature_vecs_inputs = np.load(file_path_inputs_npy)
        file_path_targets = 'C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/feature_vecs_targets.pkl'
        file_path_targets_npy = 'C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/feature_vecs_targets.npy'
        # with open(file_path_targets, 'wb') as f:
        #     torch.save(feature_vecs_targets, f)
        # np.save(file_path_targets_npi, feature_vecs_targets)
        feature_vecs_targets = np.load(file_path_targets_npy)
        np.random.seed(args.manualSeed)
        permutation = np.random.permutation(feature_vecs_inputs.shape[0])
        x_train = feature_vecs_inputs[permutation][:-int(feature_vecs_inputs.shape[0] // 5)]
        x_val = feature_vecs_inputs[permutation][-int(feature_vecs_inputs.shape[0] // 5):]
        y_train = feature_vecs_targets[permutation][:-int(feature_vecs_targets.shape[0] // 5)]
        y_val = feature_vecs_targets[permutation][-int(feature_vecs_targets.shape[0] // 5):]

        top_only_train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))

        top_only_val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.LongTensor(y_val))

        train_loader = DataLoader(top_only_train_dataset, batch_size=4096, shuffle=True)
        val_loader = DataLoader(top_only_val_dataset, batch_size=4096, shuffle=True)
        assert os.path.isfile(args.pretrained_kaggle), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load(args.pretrained_kaggle)
        # model.load_state_dict(checkpoint['state_dict'])
        classifierModel = DropClassifier()
        classifierModel.load_state_dict(torch.load(args.pretrained_kaggle), strict=False)
        classifierModel.last_linear.load_state_dict(list(model.children())[-1].state_dict())
        # del model
        optimizer = optim.Adam(classifierModel.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, verbose=True, factor=args.lr_reduce)
        classifierModel.cuda()
    else:
        validation_split = 0.2
        dataset_size = len(trainODDataset)
        if args.debug_mode == 1:
            indices = list(range(int(dataset_size / 18)))
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

    if args.evaluate == 1:
        print('\nEvaluation only')
        logger.info('\nEvaluation only')
        if args.ensemble == 1:
            val_loss, val_acc = val_ensemble(val_loader, model, criterion, use_cuda, scheduler, model_a, model_b)
        else:
            val_loss, val_acc = val(val_loader, model, criterion, use_cuda, scheduler)
        print("Val Loss: {}, Val Acc: {}".format(val_loss, val_acc))
        logger.info("Val Loss: {}, Val Acc: {}".format(val_loss, val_acc))
        test(test_loader, model, use_cuda, df_test, vis_file_out)
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # notes_logger.log("Epoch: {} of {}, LR: {}".format(epoch+1, args.epochs, state['lr']))
        logger.info("Epoch: {} of {}, LR: {}".format(epoch+1, args.epochs, state['lr']))
        if args.use_saved_feature_vecs == 1:
            train_loss, train_acc = train(train_loader, classifierModel, criterion, optimizer, epoch, use_cuda)
        else:
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        print("finished train, starting test on val set")
        logger.info("finished train, starting test on val set")

        # if (epoch % 5) == 0:
        if args.ensemble == 1:
            raise RuntimeError("ensemble should only be ran in evaluation mode (evaluate=1)")
        else:
            if args.use_saved_feature_vecs == 1:
                val_loss, val_acc = val(val_loader, classifierModel, criterion, use_cuda, scheduler)
            else:
                val_loss, val_acc = val(val_loader, model, criterion, use_cuda, scheduler)
        print("learning rate: {}".format(get_learning_rate(optimizer)))
        logger.info("learning rate: {}".format(get_learning_rate(optimizer)))
        print("train loss: {}, train accuracy: {}".format(train_loss, train_acc))
        logger.info("train loss: {}, train accuracy: {}".format(train_loss, train_acc))
        print("val loss: {}, val accuracy: {}".format(val_loss, val_acc))
        logger.info("val loss: {}, val accuracy: {}".format(val_loss, val_acc))
        # log loss and accuracy

        # save model
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        filename = str(vis_file_out)
        if args.use_saved_feature_vecs == 1:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': classifierModel.state_dict(), 'acc': val_acc, 'best_val_acc': best_val_acc,
                            'optimizer': optimizer.state_dict()
                             }, is_best, epoch, checkpoint=args.checkpoint, filename=filename)
        else:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'acc': val_acc, 'best_val_acc': best_val_acc,
                             'optimizer': optimizer.state_dict()
                             }, is_best, epoch, checkpoint=args.checkpoint, filename=filename)

    # test_loss, test_acc = test(test_loader, model, criterion, use_cuda, scheduler, df_test, vis_file_out)
    print('Best val acc: '.format(best_val_acc))
    logger.info('Best val acc: '.format(best_val_acc))


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    total_loss = 0
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

        if batch_idx == 0:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = targets.clone().data.cpu().numpy()
            all_batch_outputs = outputs_clone
            all_batch_targets = targets_clone
            # all_batch_targets = targets.clone().data.cpu().numpy()
        else:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = targets.clone().data.cpu().numpy()
            all_batch_outputs = np.concatenate((all_batch_outputs, outputs_clone), axis=0)
            all_batch_targets = np.concatenate((all_batch_targets, targets_clone), axis=0)
    accuracy = calc_accuracy(all_batch_outputs, all_batch_targets)
    return total_loss, accuracy


def val(val_loader, model, criterion, cuda, scheduler):
    # switch to evaluate mode
    model.eval()
    total_loss = 0
    print("loader length: ", len(val_loader))
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
        if batch_idx == 0:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = targets.clone().data.cpu().numpy()
            all_batch_outputs = outputs_clone
            all_batch_targets = targets_clone
        else:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = targets.clone().data.cpu().numpy()
            all_batch_outputs = np.concatenate((all_batch_outputs, outputs_clone), axis=0)
            all_batch_targets = np.concatenate((all_batch_targets, targets_clone), axis=0)
    accuracy = calc_accuracy(all_batch_outputs, all_batch_targets)
    scheduler.step(total_loss / len(val_loader))
    return total_loss, accuracy


def val_ensemble(val_loader, model, criterion, cuda, scheduler, model_a, model_b):
    # switch to evaluate mode
    model_a.eval()
    model_b.eval()
    total_loss = 0
    for batch_idx, (inputs_inc, inputs_resnet, targets) in enumerate(val_loader):
        if cuda:
            inputs_inc, inputs_resnet, targets = inputs_inc.cuda(), inputs_resnet.cuda(), targets.cuda()
        # compute output

        if args.ensemble == 1:
            outputs_a = model_a(inputs_inc)
            outputs_b = model_b(inputs_resnet)
            outputs = torch.add(outputs_a, outputs_b)
            outputs = torch.div(outputs, 2)


        loss = criterion(outputs, targets)

        total_loss += loss.item()
        # classifier_real_accuracy = np.mean((outputs_sig_np >= 0.5) == targets.data.cpu().numpy())
        # prints the precision recall curve and calculates average_precision
        # targets = targets.type(FloatTensor)
        if batch_idx == 0:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = targets.clone().data.cpu().numpy()
            all_batch_outputs = outputs_clone
            all_batch_targets = targets_clone
        else:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = targets.clone().data.cpu().numpy()
            all_batch_outputs = np.concatenate((all_batch_outputs, outputs_clone), axis=0)
            all_batch_targets = np.concatenate((all_batch_targets, targets_clone), axis=0)
    accuracy = calc_accuracy(all_batch_outputs, all_batch_targets)
    scheduler.step(total_loss / len(val_loader))
    return total_loss, accuracy


def test(test_loader, model, cuda, df_test, vis_file_out):
    # switch to evaluate mode
    model.eval()
    # end = time.time()
    # should_plot = False
    print("loader length: ", len(test_loader))
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cuda:
            inputs = inputs.cuda()
        # compute output
        if args.arch == "inception":
            outputs = model(inputs)
        else:
            outputs = model(inputs)

        if batch_idx == 0:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = np.asarray(targets).copy()
            all_batch_outputs = outputs_clone
            all_batch_targets = targets_clone
        else:
            outputs_soft = F.softmax(outputs.clone(), dim=1)
            outputs_clone = outputs_soft.data.cpu().numpy()
            targets_clone = np.asarray(targets).copy()
            all_batch_outputs = np.concatenate((all_batch_outputs, outputs_clone), axis=0)
            all_batch_targets = np.concatenate((all_batch_targets, targets_clone), axis=0)

    # print("Targets: ", all_batch_targets)
    # print("Outputs: ", all_batch_outputs)
    df_pred = pd.DataFrame(all_batch_outputs, index=all_batch_targets, columns=df_test.columns)
    df_pred.index.name = 'id'
    filename = './logs/testacc' + str(vis_file_out) + '1.csv'
    df_pred.to_csv(filename)
    return


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
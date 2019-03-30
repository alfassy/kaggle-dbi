import argparse
import os
import numpy as np
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from dataset.kaggle_dataset import DBI_dataset, DBI_dataset_ensemble
from scripts.configure_logging import configure_logging, get_learning_rate
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import pandas as pd
from pathlib import Path
from scripts.testing_functions import calc_accuracy, calc_per_class_accuracy
import pretrainedmodels
from scripts.getting_features import get_feature_vecs, DropClassifier

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
# Optimization options
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=4, type=int, metavar='N',
                    help='train batchsize (default: 8)')
parser.add_argument('--test_batch', default=4, type=int, metavar='N',
                    help='test batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_reduce', default=0.3, type=float, help='lr reduce on plateau factor')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--env_name', type=str, default='DBINasnetAllPretrainedKaggleTopOnly', help='Environment name for naming purposes')
# Checkpoints
parser.add_argument('--checkpoint', default='C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/', type=str, metavar='PATH',
                    help='path to save models (default: checkpoint)')
parser.add_argument('--resume', default='C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/Best_model.pkl',
                    type=str, metavar='PATH', help='path to latest full model for continuing a full model run (default: none)')
parser.add_argument('--resume_top_module', default='',
                    type=str, metavar='PATH', help='path to load top module (classifier only) (default: none)')

# Architecture
parser.add_argument('--transfer_oxford', default='',
                    type=str, metavar='PATH', help='path to model pretrained on oxford pets model (default: none)')
parser.add_argument('--pretrained_kaggle', default='',
                    type=str, metavar='PATH', help='path to base model pretrained on kaggle, only works with  use_saved_feature_vecs==1 (default: none)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnet',
                    help='model architecture:(resnet18/ inception/ nasnet)')
# Miscs
parser.add_argument('--train_path', default='./data/kaggle_dbi/train', type=str, metavar='PATH', help='path to train kaggle image data')
parser.add_argument('--train_labels_path', default='./data/kaggle_dbi/labels.csv', type=str, metavar='PATH', help='path to train kaggle labels data')
parser.add_argument('--test_path', default='./data/kaggle_dbi/test', type=str, metavar='PATH', help='path to test kaggle image data')
parser.add_argument('--file_path_inputs_npy_to_load',
                    default='',
                    type=str, metavar='PATH', help='path to saved feature vecs')
parser.add_argument('--file_path_targets_npy_to_load',
                    default='',
                    type=str, metavar='PATH', help='path to saved targets')

parser.add_argument('--manualSeed', type=int, help='manual seed for random function')
parser.add_argument('--debug_mode', type=int, default=0, help='activate fast debug mode?(default: 0) 0(no)/1(yes)')

parser.add_argument('--freezeLayers', type=str, default='all',
                    help='how many layers should we freeze? all is all except last layer, half is the number of freezeLayersNum options:none/half/all')
parser.add_argument('--freezeLayersNum', type=int, default=19,
                    help='how many layers should we freeze? options:int')

parser.add_argument('--oxford_augment', type=int, default=1, help='add oxfords images for training 0(no)/1(yes)')
parser.add_argument('--print_per_class_acc', type=int, default=1, help='print per class accuracy 0(no)/1(yes)')
parser.add_argument('--evaluate', type=int, default=1, help='evaluate (ONLY!) model on validation and test sets? 0(no)/1(yes)')
parser.add_argument('--use_saved_feature_vecs', type=int, default=0,
                    help='use Saved feature vecs for training data for better performance in long runs. if file_path_(inputs/tagets)_npy_to_load isnt given then creates such files 0(no)/1(yes)')
parser.add_argument('--random_angle', type=float, default=10, help='Angle of random augmentation.')
parser.add_argument('--ensemble', type=int, default=0, help='should we use model ensemble? does not support "use_saved_feature_vecs==1" (no)/1(yes)')
parser.add_argument('--ensemble_path_a', default='C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/DBINasnet19Oxford2019;3;28;22;58best.pkl', type=str, metavar='PATH', help='path to 1st ensemble model (default: none)')
parser.add_argument('--ensemble_path_b', default='C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/DBINasnet192019;3;28;23;4best.pkl', type=str, metavar='PATH', help='path to 2nd ensemble model (default: none)')

#Device options
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

# check if cuda is available
use_cuda = torch.cuda.is_available()

# set random seed
if args.manualSeed is None:
    # args.manualSeed = random.randint(1, 10000)
    args.manualSeed = 5
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# initialiaze global perfomance meters
best_val_loss = 100000  # best test loss
best_val_acc = 0   # best test accuracy

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def main():
    global best_val_loss
    global best_val_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    env_name = args.env_name
    # create a unique id for naming purposes
    now = datetime.datetime.now()
    vis_file_out = env_name + str(now.year) + ';' + str(now.month) + ';' + \
                   str(now.day) + ';' + str(now.hour) + ';' + str(now.minute)
    # set up logger
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
        trainODDataset = DBI_dataset(fp_train, df_train, df_test, args.oxford_augment, is_train=True, transform=train_transform2)
        testODDataset = DBI_dataset(fp_test, df_train, df_test, args.oxford_augment, is_train=False, transform=test_transform)

    train_loader_all = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch, shuffle=False)
    test_indices = list(range(args.test_batch))
    test_sampler = SubsetRandomSampler(test_indices)
    if args.debug_mode == 1:
        test_loader = torch.utils.data.DataLoader(testODDataset, batch_size=args.test_batch, sampler=test_sampler,
                                                  shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(testODDataset, batch_size=args.test_batch, shuffle=False)

    # load pretrained model for a soft start
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
        raise NotImplementedError("only inception_v3, nanset and resnet18 are available at the moment")
    # set up loss function
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

    # move models to GPU
    if use_cuda:
        print("using gpu")
        model.cuda()
        criterion.cuda()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
    optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True, factor=args.lr_reduce)

    # do model ensemble
    if args.ensemble == 1:
        print('==> doing ensemble..')
        logger.info('==> doing ensemble..')
        model_b = pretrainedmodels.nasnetalarge(num_classes=1000, pretrained='imagenet')
        num_ftrs = model_b.last_linear.in_features
        model_b.last_linear = nn.Linear(num_ftrs, out_size)
        assert os.path.isfile(args.ensemble_path_a), 'Error: no checkpoint directory found!'
        checkpointa = torch.load(args.ensemble_path_a)
        checkpointb = torch.load(args.ensemble_path_b)
        model.load_state_dict(checkpointa['state_dict'])
        model_b.load_state_dict(checkpointb['state_dict'])
        model.cuda()
        model_b.cuda()

    # continue training of a model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_val_loss = checkpoint['best_val_loss']
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
            special_child = args.freezeLayersNum
        elif args.arch == 'resnet18':
            special_child = args.freezeLayersNum
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

    # save feature vecs for better runtime on long epochs training
    if args.use_saved_feature_vecs == 1:
        args.epochs = 10000
        # load base model pretrained on kaggle into memory
        if args.pretrained_kaggle:
            checkpoint = torch.load(args.pretrained_kaggle)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
        # load saved feature vectors and labels
        if args.file_path_inputs_npy_to_load:
            feature_vecs_inputs = np.load(args.file_path_inputs_npy_to_load)
            feature_vecs_targets = np.load(args.file_path_targets_npy_to_load)
        else:
            # if no saved feature vectors and labels are given then create those files for future use
            feature_vecs_inputs, feature_vecs_targets = get_feature_vecs(train_loader_all, model)
            file_path_inputs_npy_to_save = 'C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/feature_vecs_inputs' + str(vis_file_out) + '.npy'
            np.save(file_path_inputs_npy_to_save, feature_vecs_inputs)
            file_path_targets_npy_to_save = 'C:/Users/Alfassy/PycharmProjects/Dog_Breed_Identification/saved_models/kaggle_dbi/feature_vecs_targets' + str(vis_file_out) + '.npy'
            np.save(file_path_targets_npy_to_save, feature_vecs_targets)

        # separate data to train and val
        validation_split = 0.2
        # num without oxford
        dataset_size = trainODDataset.img_num
        if args.debug_mode == 1:
            indices = list(range(int(dataset_size / 18)))
        else:
            indices = list(range(dataset_size))
        split = int(np.floor(validation_split * len(indices)))
        np.random.seed(args.manualSeed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        x_train = feature_vecs_inputs[train_indices]
        x_val = feature_vecs_inputs[val_indices]
        y_train = feature_vecs_targets[train_indices]
        y_val = feature_vecs_targets[val_indices]
        # the dataset is divided to [kaggle, oxford] so if we wish, we'll add oxford to the train data only.
        if args.oxford_augment == 1:
            x_train = np.concatenate((x_train, feature_vecs_inputs[range(trainODDataset.img_num, trainODDataset.img_num_with_oxford)]))
            y_train = np.concatenate((y_train, feature_vecs_targets[range(trainODDataset.img_num, trainODDataset.img_num_with_oxford)]), axis=0)

        top_only_train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
        top_only_val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.LongTensor(y_val))
        train_loader = DataLoader(top_only_train_dataset, batch_size=4096, shuffle=True)
        val_loader = DataLoader(top_only_val_dataset, batch_size=4096, shuffle=True)
        # initialize a top model consists of a fc and dropout
        classifierModel = DropClassifier()
        # if starting from a trained base_model, also load his FC layer as an initialization unless resume_top_module is given
        if args.pretrained_kaggle:
            assert os.path.isfile(args.pretrained_kaggle), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(args.pretrained_kaggle)

            # classifierModel.last_linear.load_state_dict(list(model.children())[-1].state_dict())
            classifierModel = DropClassifier(checkpoint['state_dict']['last_linear.weight'], checkpoint['state_dict']['last_linear.bias'])
            # classifierModel.last_linear.weight.data.fill_(checkpoint['state_dict']['last_linear.weight'])
            # classifierModel.last_linear.bias.data = checkpoint['state_dict']['last_linear.bias']
            print(torch.load(args.pretrained_kaggle)['state_dict'])
            print(classifierModel.last_linear.weight)
            del checkpoint
        # load saved top module
        if args.resume_top_module:
            classifierModel.load_state_dict(torch.load(args.resume_top_module)['state_dict'])
        # set up optimizer for top module
        optimizer = optim.Adam(classifierModel.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, verbose=True, factor=args.lr_reduce)
        classifierModel.cuda()
    else:
        # split train and val
        validation_split = 0.2
        dataset_size = trainODDataset.img_num
        if args.debug_mode == 1:
            indices = list(range(int(dataset_size / 18)))
        else:
            indices = list(range(dataset_size))
        split = int(np.floor(validation_split * len(indices)))
        np.random.seed(args.manualSeed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # the dataset is divided to [kaggle, oxford] so if we wish, we'll add oxford to the train data only.
        if args.oxford_augment == 1:
            train_indices = np.concatenate((train_indices, range(trainODDataset.img_num, trainODDataset.img_num_with_oxford)), axis=0)
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch,
                                                   sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(trainODDataset, batch_size=args.train_batch,
                                                 sampler=valid_sampler)
    # evaluation only mode
    if args.evaluate == 1:
        print('\nEvaluation only')
        logger.info('\nEvaluation only')
        # not checking val loss and acc in ensemble mode
        if args.ensemble == 1:
                test(test_loader, model, model_b, None, use_cuda, df_test, vis_file_out)
        else:
            # check val accuracy
            if args.use_saved_feature_vecs == 1:
                val_loss, val_acc, class_acc_list = val(val_loader, classifierModel, criterion, use_cuda, scheduler)
            else:
                val_loss, val_acc, class_acc_list = val(val_loader, model, criterion, use_cuda, scheduler)
            print("Val Loss: {}, Val Acc: {}".format(val_loss, val_acc))
            logger.info("Val Loss: {}, Val Acc: {}".format(val_loss, val_acc))
            if args.print_per_class_acc:
                for class_ind, class_acc in enumerate(class_acc_list):
                    print('Class {} accuracy: {}'.format(class_ind, class_acc))
            # create copmpetition excel
            if args.use_saved_feature_vecs == 1:
                # feature_vecs_inputs, feature_vecs_targets = get_feature_vecs(test_loader, model, test_mode=True)
                # print(feature_vecs_targets)
                # top_only_test_dataset = DBI_dataset_test_time(torch.FloatTensor(feature_vecs_inputs), feature_vecs_targets)
                # test_loader = DataLoader(top_only_test_dataset, batch_size=4096, shuffle=False)
                test(test_loader, model, None, classifierModel, use_cuda, df_test, vis_file_out)
            else:
                test(test_loader, model, None, None, use_cuda, df_test, vis_file_out)
        return

    # Train and val main loop
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        logger.info("Epoch: {} of {}, LR: {}".format(epoch+1, args.epochs, state['lr']))
        if args.use_saved_feature_vecs == 1:
            train_loss, train_acc = train(train_loader, classifierModel, criterion, optimizer, use_cuda)
        else:
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, use_cuda)
        print("finished train, starting test on val set")
        logger.info("finished train, starting test on val set")
        if args.ensemble == 1:
            raise RuntimeError("ensemble should only be ran in evaluation mode (evaluate=1)")
        else:
            if args.use_saved_feature_vecs == 1:
                val_loss, val_acc, class_acc_list = val(val_loader, classifierModel, criterion, use_cuda, scheduler)
            else:
                val_loss, val_acc, class_acc_list = val(val_loader, model, criterion, use_cuda, scheduler)
        print("learning rate: {}".format(get_learning_rate(optimizer)))
        logger.info("learning rate: {}".format(get_learning_rate(optimizer)))
        print("train loss: {}, train accuracy: {}".format(train_loss, train_acc))
        logger.info("train loss: {}, train accuracy: {}".format(train_loss, train_acc))
        print("val loss: {}, val accuracy: {}".format(val_loss, val_acc))
        logger.info("val loss: {}, val accuracy: {}".format(val_loss, val_acc))
        if args.print_per_class_acc:
            for class_ind, class_acc in enumerate(class_acc_list):
                print('Class {} accuracy: {}'.format(class_ind, class_acc))
        # save model
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        best_val_acc = max(val_acc, best_val_acc)
        filename = str(vis_file_out)
        if args.use_saved_feature_vecs == 1:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': classifierModel.state_dict(), 'loss': val_loss, 'best_val_loss': best_val_loss,
                            'optimizer': optimizer.state_dict()
                             }, is_best, checkpoint=args.checkpoint, filename=filename)
        else:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'acc': val_loss, 'best_val_loss': best_val_loss,
                             'optimizer': optimizer.state_dict()
                             }, is_best, checkpoint=args.checkpoint, filename=filename)

    # test_loss, test_acc = test(test_loader, model, criterion, use_cuda, scheduler, df_test, vis_file_out)
    print('Best val acc: {}'.format(best_val_acc))
    print('Best val loss: {}'.format(best_val_loss))
    logger.info('Best val acc: {}'.format(best_val_acc))
    logger.info('Best val loss: {}'.format(best_val_loss))


def train(train_loader, model, criterion, optimizer, use_cuda):
    '''
    Training function for a full epoch
    :param train_loader: data loader of train data
    :param model: model to train
    :param criterion: loss function
    :param optimizer: GD optimizer
    :param use_cuda: bool, True if gpu is available
    :return:
    total loss: float, total training loss over the entire epoch
    accuracy: float, average classification accuracy over the entire epoch
    '''
    # switch to train mode
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # compute output
        if args.arch == "inception":
            outputs, aux_logits = model(inputs)
            loss = criterion(outputs, targets)
            # loss2 = criterion(aux_logits, targets)
            # loss = (loss1 + loss2) / 2
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        # record loss
        total_loss += loss.item()
        # compute gradient and do gradient step step
        loss.backward()
        optimizer.step()
        # save whole data for accuracy calculations
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
    # measure accuracy
    accuracy = calc_accuracy(all_batch_outputs, all_batch_targets)
    return total_loss, accuracy


def val(val_loader, model, criterion, cuda, scheduler):
    '''
    Validation test function
    :param val_loader: data loader of validation data
    :param model: model to test with
    :param criterion: loss function
    :param use_cuda:bool,  True if gpu is available
    :param scheduler: learning rate scheduler
    :return:
    total_loss: float, total validation loss over the entire epoch
    accuracy: float, average classification accuracy over the entire epoch
    per_class_accuracy: list, list with average classification accuracy for each class over the entire epoch
    '''
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
        # record loss
        total_loss += loss.item()
        # save whole data for accuracy calculations
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
    # measure accuracy
    accuracy = calc_accuracy(all_batch_outputs, all_batch_targets)
    if args.print_per_class_acc:
        per_class_accuracy = calc_per_class_accuracy(all_batch_outputs, all_batch_targets)
    else:
        per_class_accuracy = None
    # do learning rate scheduler step
    scheduler.step(total_loss / len(val_loader))
    return total_loss, accuracy, per_class_accuracy


def test(test_loader, model_a, model_b, classifier, cuda, df_test, vis_file_out):
    '''
    Create the kagggle excel result submission file over the test data
    :param test_loader: data loader of test data
    :param model: model to test with, if args.use_saved_feature_vecs == 1 then this should be the base_model for feature extraction
    :param classifier: top module which classify on top of a feature extractor
    :param cuda: bool, True if gpu is available
    :param df_test: pandas dataframe, sample_submission file for creating our own submission
    :param vis_file_out: str unique, string for naming purposes
    :return: None
    '''
    # switch to evaluate mode
    model_a.eval()
    print("test loader length: ", len(test_loader))
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cuda:
            inputs = inputs.cuda()
        # compute output
        if args.use_saved_feature_vecs == 1:
            outputs = model_a.features(inputs)
            outputs = model_a.avg_pool(outputs)
            outputs = outputs.view(outputs.size(0), -1)
            outputs_a = classifier(outputs)
        elif args.ensemble == 1:
            outputs_a = model_a(inputs)
            outputs_b = model_b(inputs)
        else:
            outputs_a = model_a(inputs)

        # save whole data for accuracy calculations
        if batch_idx == 0:
            outputs_soft_a = F.softmax(outputs_a.clone(), dim=1)
            if args.ensemble == 1:
                outputs_soft_b = F.softmax(outputs_b.clone(), dim=1)
                outputs_clone = torch.add(outputs_soft_a, outputs_soft_b)
                outputs_clone = torch.div(outputs_clone, 2)
                outputs_clone = outputs_clone.data.cpu().numpy()
            else:
                outputs_clone = outputs_soft_a.data.cpu().numpy()
            targets_clone = np.asarray(targets).copy()
            all_batch_outputs = outputs_clone
            all_batch_targets = targets_clone
        else:
            outputs_soft_a = F.softmax(outputs_a.clone(), dim=1)
            if args.ensemble == 1:
                outputs_soft_b = F.softmax(outputs_b.clone(), dim=1)
                outputs_clone = torch.add(outputs_soft_a, outputs_soft_b)
                outputs_clone = torch.div(outputs_clone, 2).data.cpu().numpy()
            else:
                outputs_clone = outputs_soft_a.data.cpu().numpy()
            targets_clone = np.asarray(targets).copy()
            all_batch_outputs = np.concatenate((all_batch_outputs, outputs_clone), axis=0)
            all_batch_targets = np.concatenate((all_batch_targets, targets_clone), axis=0)

    # generate competition excel and save it
    df_pred = pd.DataFrame(all_batch_outputs, index=all_batch_targets, columns=df_test.columns)
    df_pred.index.name = 'id'
    filename = './logs/testacc' + str(vis_file_out) + '1.csv'
    df_pred.to_csv(filename)
    return


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    '''
    save model for future use
    :param state: dictionary, dictionary of run data to save
    :param is_best: bool, true if current epoch has the lowest loss so far
    :param checkpoint: str, folder to save data to
    :param filename: str, unique name for naming purposes
    :return: None
    '''
    file_path = checkpoint + filename + 'last' + '.pkl'
    with open(file_path, 'wb') as f:
        torch.save(state, f)
    # if best loss so far also save as best model
    if is_best:
        best_file_path = checkpoint + filename + 'best' + '.pkl'
        with open(best_file_path, 'wb') as f:
            torch.save(state, f)
    return


if __name__ == '__main__':
    main()

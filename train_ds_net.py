import random
import os
import sys
import logging
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import datetime
import numpy as np
import attack_generator as attack
from attack_generator import *
from earlystop import *
from model.ds_net import *

from utils import count_parameters_in_MB, Logger
import utils
import time
import glob
import math
from torch.autograd import Variable

# apex
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
amp.register_float_function(torch, 'sigmoid')



parser = argparse.ArgumentParser(
    description='PyTorch Adversarial Training')
parser.add_argument('--epochs', type=int, default=120,
                    metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd',
                    default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='SGD momentum')
parser.add_argument('--seed', type=int, default=0,
                    metavar='S', help='random seed')
parser.add_argument('--rand_init', type=bool, default=True,
                    help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001,
                    help="random sample parameter for adv data generation")
parser.add_argument('--resume', type=str, default='',
                    help='whether to resume training, default: None')
parser.add_argument("--classes", type=int, default=10)
parser.add_argument("--note", type=str, default='')

# training config.
parser.add_argument('--adv_train', type=int, default=1)
parser.add_argument('--arch_learning_rate', type=float, default=1e-3)
parser.add_argument('--arch_weight_decay', type=float, default=1e-3)
parser.add_argument('--use_subset', type=int, default=0)
parser.add_argument('--is_at', type=int, default=1)
parser.add_argument('--is_trades', type=int, default=0)
parser.add_argument('--is_mart', type=int, default=0)
parser.add_argument('--trades_beta', type=float, default=6.)
parser.add_argument('--mart_beta', type=float, default=6.)
parser.add_argument('--batch_size', default=128, type=int)

# attack config.
parser.add_argument('--epsilon', type=float, default=0.031,
                    help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10,
                    help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help="choose from cifar10,svhn")

# network config.
parser.add_argument("--norm", type=str, default="b")
parser.add_argument("--init_channel", type=int, default=20)
parser.add_argument("--depth", type=int, default=15)
parser.add_argument("--block", type=int, default=1)
parser.add_argument('--num_op', type=int, default=4)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--factor', type=int, default=4)

# variants.
parser.add_argument('--space_version', type=str, default='v1') # v2 can further improve acc.
parser.add_argument('--is_softmax', type=int, default=1)
parser.add_argument('--is_fixed', type=int, default=0)
parser.add_argument('--start_evaluation', type=int, default=0)
parser.add_argument('--use_amp', type=int, default=0)


# distribution args.
parser.add_argument('--is_normal', type=int, default=0)
parser.add_argument('--is_uniform', type=int, default=0)
parser.add_argument('--is_log_normal', type=int, default=0)
parser.add_argument('--is_exponential', type=int, default=0)
parser.add_argument('--is_geometric', type=int, default=0)
parser.add_argument('--is_trained', type=int, default=0)


args = parser.parse_args()
print(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.set_device(int(args.gpu))
# training settings
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(args.seed)

# different adv training methods.
if args.dataset == 'cifar10':
    if args.is_trades:
        args.epochs = 85
    elif args.is_mart:
        args.epochs = 90
    else:
        args.epochs = 120
else:
    if args.is_trades:
        args.epochs = 65
    elif args.is_mart:
        args.epochs = 70
    else:
        args.epochs = 100

if not args.adv_train:
    args.is_at = 0
    args.is_trades = 0
    args.is_mart = 0


if args.is_trades:
    args.is_at = 0
    args.is_mart = 0

if args.is_mart:
    args.is_at = 0
    args.is_trades = 0
    args.omega = 0


if args.is_at:
    args.is_trades = 0
    args.is_mart = 0


if args.dataset == 'svhn':
    args.step_size = args.epsilon / 10.
    # args.lr = 0.05


if args.adv_train:
    args.save = '{}-{}-seed-{}-adv_train-{}-softmax-{}-fixed-{}-epsilon-{}-step_size-{}-init_channel-{}-at-{}-trades-{}-mart-{}-beta-{}-{}'.format(
                                                            args.note,
                                                            args.dataset,
                                                            str(args.seed),
                                                            str(args.adv_train),
                                                            str(args.is_softmax),
                                                            str(args.is_fixed),
                                                            str(args.epsilon),
                                                            str(args.step_size),
                                                            str(args.init_channel),
                                                            str(args.is_at),
                                                            str(args.is_trades),
                                                            str(args.is_mart),
                                                            str(args.trades_beta),
                                                            time.strftime("%Y%m%d-%H%M%S"))
else:
    args.save = '{}-{}-seed-{}-adv_train-{}-softmax-{}-fixed-{}-init_channel-{}-{}'.format(
                                                            args.note,
                                                            args.dataset,
                                                            str(args.seed),
                                                            str(args.adv_train),
                                                            str(args.is_softmax),
                                                            str(args.is_fixed),
                                                            str(args.init_channel),
                                                            time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
out_dir = args.save

# specially for logging the arch_params.
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(out_dir, 'log_arch_param.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def TRADES_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo TREADES: https://github.com/yaodongyu/TRADES
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
    loss_natural = nn.CrossEntropyLoss(
        reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                    F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def MART_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo MART https://github.com/YisenWang/MART
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(target)
    adv_probs = F.softmax(adv_logits, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(adv_logits, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(natural_logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss




def train(args,
          epoch,
          model,
          train_loader,
          optimizer,
          optimizer_arch,
          scheduler,
          logger_loss):
    #print(global_noise_data)
    starttime = datetime.datetime.now()
    loss_sum = 0
    bp_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # Get adversarial training data via PGD
        if args.adv_train:
            if args.is_trades:
                output_adv = pgd_train(
                                args,
                                model,
                                optimizer,
                                data, target,
                                args.epsilon,
                                args.step_size,
                                args.num_steps,
                                'kl',
                                'trades',
                                args.rand_init,
                                args.omega,
                                state='train')
                bp_count += args.num_steps * args.batch_size
            elif args.is_at or args.is_mart:
                output_adv = pgd_train(
                                args,
                                model,
                                optimizer,
                                data, target,
                                args.epsilon,
                                args.step_size,
                                args.num_steps,
                                'cent',
                                'Madry',
                                args.rand_init,
                                args.omega,
                                state='train')
                bp_count += args.num_steps * args.batch_size
        else:
            output_adv = data
            bp_count += 0
        model.train()
        if args.is_softmax:
            model.arch_param.requires_grad = True

        optimizer.zero_grad()
        if args.is_softmax:
            optimizer_arch.zero_grad()
        output = model(output_adv)
        # calculate adversarial training loss
        if args.adv_train and args.is_trades:
            adv_logits = output
            natural_logits = model(data)
            loss = TRADES_loss(adv_logits, natural_logits,
                            target, args.trades_beta)
        elif args.adv_train and args.is_mart:
            adv_logits = output
            natural_logits = model(data)
            loss = MART_loss(adv_logits, natural_logits,
                            target, args.mart_beta)
        else:
            loss = nn.CrossEntropyLoss(reduction='mean')(output, target)
        loss_sum = loss_sum + loss.item()
        if args.use_amp:
            if args.is_softmax:
                with amp.scale_loss(loss, [optimizer, optimizer_arch]) as scaled_loss: 
                    scaled_loss.backward()
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss: 
                    scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        if args.is_softmax:
            optimizer_arch.step()

    print('loss this epoch: ', loss_sum / (batch_idx + 1))
    logger_loss.append([epoch + 1, loss_sum / (batch_idx + 1)])
    bp_count_avg = bp_count / len(train_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    if args.is_softmax:
        print('arch_param this epoch is: ', model.arch_param)
        logging.info(epoch)
        logging.info(model.arch_param)

    return time, loss_sum, bp_count_avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if args.dataset == 'svhn':
        if epoch <= 5:
            lr = args.lr * epoch / 5.
        if not args.is_trades:
            if epoch >= 40:
                lr = args.lr * 0.1
            if epoch >= 70:
                lr = args.lr * 0.01
            if epoch >= 90:
                lr = args.lr * 0.005
        else:
            if epoch >= 55:
                lr = args.lr * 0.1
            if epoch >= 70:
                lr = args.lr * 0.01
    else:
        # if epoch <= 5:
        #     lr = args.lr * epoch / 5.
        if not args.is_trades:
            if epoch >= 60:
                lr = args.lr * 0.1
            if epoch >= 90:
                lr = args.lr * 0.01
            if epoch >= 110:
                lr = args.lr * 0.005
        else:
            if epoch >= 75:
                lr = args.lr * 0.1
            if epoch >= 90:
                lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('current lr: ', lr)



def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root='~/data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=True, transform=transform_test)
    if args.use_subset:
        testset = torch.utils.data.Subset(testset,
                                          random.sample(range(len(testset)), 1280))
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(
        root='~/data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    testset = torchvision.datasets.SVHN(
        root='~/data', split='test', download=True, transform=transform_test)
    # import ipdb; ipdb.set_trace()
    if args.use_subset:
        testset = torch.utils.data.Subset(testset,
                                          random.sample(range(len(testset)), 2600))
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

print('==> Load Model')


model = CIFAR10_DS_NET(args=args,
                                    layers=args.block,
                                    num_classes=args.classes,
                                    init_channel=args.init_channel,
                                    norm_type=args.norm,
                                    downsample_type="r",
                                    a21=0.25,
                                    b10=1.0,
                                    a_logic=False,
                                    b_logic=True).cuda()
# import ipdb; ipdb.set_trace()
# model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
if args.is_softmax:                                               
    arch_parameters = model.arch_param
    optimizer_arch = torch.optim.Adam([arch_parameters],
                                lr=args.arch_learning_rate,
                                betas=(0.5, 0.999),
                                weight_decay=args.arch_weight_decay)
    if args.use_amp:
        model, [optimizer, optimizer_arch] = amp.initialize(model, [optimizer, optimizer_arch], 
        opt_level="O1", loss_scale=1.0)
else:
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1.0)


start_epoch = 0
# Resume
title = 'PGD train'
if args.resume:
    # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
    print('==> Adversarial Training Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(
        out_dir, 'log_results.txt'), title=title, resume=True)
    logger_loss = Logger(os.path.join(
        out_dir, 'log_loss.txt'), title=title, resume=True)
else:
    print('==> Adversarial Training')
    logger_test = Logger(os.path.join(
        out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc',
                           'FGSM Acc', 'PGD20 Acc',
                           'PGD20-double Acc', 'PGD100 Acc', 'CW Acc'])
    logger_loss = Logger(os.path.join(
        out_dir, 'log_loss.txt'), title=title)
    logger_loss.set_names(['Epoch', 'loss'])

test_nat_acc = 0
fgsm_acc = 0
test_pgd20_acc = 0
cw_acc = 0
best_epoch = 0
test_pgd20_acc_double = 0
test_pgd100_acc = 0
test_aa_acc = 0

# logger_test.append(model)
with open(os.path.join(
        out_dir, 'model_param.txt'), 'w') as file:
    file.write("param size = " +
               str(count_parameters_in_MB(model)) + "MB" + '\n')
print("param size = " + str(count_parameters_in_MB(model)) + "MB" + '\n')

best_metric = 0
for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch + 1)
    if args.is_softmax:
        train_time, train_loss, bp_count_avg = train(
            args, epoch, model,
            train_loader, optimizer, optimizer_arch, None,
            logger_loss)
    else:
        train_time, train_loss, bp_count_avg = train(
            args, epoch, model,
            train_loader, optimizer, None, None,
            logger_loss)

    if epoch >= args.start_evaluation:
        model.eval()
        if args.is_softmax:
            model.arch_param.requires_grad = False
        # Evalutions the same as FAT.
        if args.dataset == 'cifar10':
            evaluate_epsilon = 0.031
        else:
            evaluate_epsilon = 0.031
        if args.is_trades:
            step_size = 0.003
        else:
            step_size = 0.031 / 4
        loss, test_nat_acc = attack.eval_clean(args, model, test_loader)
        # aa is time consuming to evaluate. consider commenting it out.
        loss, test_aa_acc = attack.eval_robust_aa(
            args,
            model, 
            optimizer,
            test_loader,
            perturb_steps=20,
            epsilon=evaluate_epsilon,
            step_size=step_size,
            loss_fn="cent",
            category="Madry",
            rand_init=True)
        print(test_aa_acc)

        loss, fgsm_acc = attack.eval_robust(
            args,
            model, 
            optimizer,
            test_loader,
            perturb_steps=1,
            epsilon=evaluate_epsilon,
            step_size=evaluate_epsilon,
            loss_fn="cent",
            category="Madry",
            rand_init=True)
            #arch_param_test=arch_param_test)
        loss, test_pgd20_acc = attack.eval_robust(
            args,
            model, 
            optimizer,
            test_loader,
            perturb_steps=20,
            epsilon=evaluate_epsilon,
            step_size=step_size,
            loss_fn="cent",
            category="Madry",
            rand_init=True)
            #arch_param_test=arch_param_test)
        # loss, test_pgd20_acc_double = attack.eval_robust(
        #     args,
        #     model, test_loader,
        #     perturb_steps=20,
        #     epsilon=evaluate_epsilon * 2,
        #     step_size=step_size,
        #     loss_fn="cent",
        #     category="Madry",
        #     rand_init=True,
        #     arch_param_test=arch_param_test)
        test_pgd20_acc_double = 0.0
        # loss, test_pgd100_acc = attack.eval_robust(
        #     args,
        #     model, test_loader,
        #     perturb_steps=100,
        #     epsilon=evaluate_epsilon,
        #     step_size=step_size,
        #     loss_fn="cent",
        #     category="Madry",
        #     rand_init=True,
        #     arch_param_test=arch_param_test)
        test_pgd100_acc = 0.0
        loss, cw_acc = attack.eval_robust(
            args,
            model, 
            optimizer,
            test_loader,
            perturb_steps=30,
            epsilon=evaluate_epsilon,
            step_size=step_size,
            loss_fn="cw",
            category="Madry",
            rand_init=True)
        

        print(
            'Epoch: [%d | %d] | Train Time: %.2f s | BP Average: %.2f | Natural Test Acc %.2f | FGSM Test Acc %.2f | PGD20 Test Acc %.2f | PGD20-double Test Acc %.2f | PGD100 Test Acc %.2f | CW Test Acc %.2f |\n' % (
                epoch + 1,
                args.epochs,
                train_time,
                bp_count_avg,
                test_nat_acc,
                fgsm_acc,
                test_pgd20_acc,
                test_pgd20_acc_double,
                test_pgd100_acc,
                cw_acc)
        )

        logger_test.append(
            [epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc,
             test_pgd20_acc_double, test_pgd100_acc, cw_acc])
        current_metric = test_pgd20_acc
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'bp_avg': bp_count_avg,
            'test_nat_acc': test_nat_acc,
            'fgsm_acc': fgsm_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'cw_acc': cw_acc,
            'test_pgd20_acc_double': test_pgd20_acc_double,
            'test_pgd100_acc': test_pgd100_acc,
            'optimizer': optimizer.state_dict(),
            },filename='checkpoint_best.pth.tar')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'bp_avg': bp_count_avg,
            'test_nat_acc': test_nat_acc,
            'fgsm_acc': fgsm_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'cw_acc': cw_acc,
            'test_pgd20_acc_double': test_pgd20_acc_double,
            'test_pgd100_acc': test_pgd100_acc,
            'optimizer': optimizer.state_dict(),
        })
    else:
        print('epoch: ', epoch + 1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'bp_avg': bp_count_avg,
            'optimizer': optimizer.state_dict()})

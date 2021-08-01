import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# from models import *
from autoattack  import AutoAttack
from model.ds_net import *
from apex import amp, optimizers
from torchvision import transforms
import random
from model.wideresnet import *
from model.wrn_madry import *


def cwloss(output, target, confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence,
                        min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def pgd_train(
        args,
        model,
        optimizer,
        data,
        target,
        epsilon,
        step_size,
        num_steps,
        loss_fn,
        category,
        rand_init,
        omega,
        state='train'):
    model.eval()
    if args.is_softmax:
        model.arch_param.requires_grad = False

    if category == "trades":
        x_adv = data.detach() + 0.001 * \
            torch.randn(data.shape).cuda().detach(
        ) if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon,
                                                                   data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                output = model(x_adv)
                loss_adv = nn.CrossEntropyLoss(
                    reduction="mean")(output, target)
            if loss_fn == "cw":
                output = model(x_adv)
                loss_adv = cwloss(output, target)
            if loss_fn == "kl":
                output = model(x_adv)
                output_nat = model(data)
                criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
                loss_adv = criterion_kl(F.log_softmax(
                    output, dim=1),
                    F.softmax(output_nat, dim=1))
        if not args.use_amp:
            loss_adv.backward()#retain_graph=True
        else:
            with amp.scale_loss(loss_adv, optimizer) as scaled_loss:
                scaled_loss.backward()
        eta = step_size * x_adv.grad.data.sign()
        if state == 'train':
            x_adv = x_adv.detach() + eta + omega * torch.randn(x_adv.shape).detach().cuda()
        else:
            x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def eval_clean(args, model, test_loader):
    model.eval()
    if args.is_softmax:
        model.arch_param.requires_grad = False
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(
                reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def AA_data_generate(args, data, target, model, optimizer, epsilon, batch_size=128):
    adversary = AutoAttack(
                model, norm='Linf', eps=epsilon, version='standard')
    adv_complete = adversary.run_standard_evaluation(
        data, target, bs=batch_size)
    return adv_complete


def eval_robust(args,
                model, 
                optimizer,
                test_loader, 
                perturb_steps, 
                epsilon, 
                step_size, 
                loss_fn, 
                category, 
                rand_init):
    model.eval()
    if args.is_softmax:
        model.arch_param.requires_grad = False
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = pgd_train(args, model, optimizer, data, target, epsilon, step_size,
                        perturb_steps, loss_fn, category, rand_init, 0.001,
                        state='test')
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(
                reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Loss_fn:{}, Perturb steps:{}, Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(loss_fn, perturb_steps, epsilon, step_size,
                                                                                                                                                                test_loss, correct, len(
                                                                                                                                                                    test_loader.dataset),
                                                                                                                                                                100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_robust_aa(args,
                model, 
                optimizer,
                test_loader, 
                perturb_steps, 
                epsilon, 
                step_size, 
                loss_fn, 
                category, 
                rand_init):
    model.eval()
    if args.is_softmax:
        model.arch_param.requires_grad = False
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = AA_data_generate(args, data, target, model, optimizer, epsilon)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(
                reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Type:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format('AUTOATTACK', test_loss, correct, len(test_loader.dataset),
                                                                                                                                                                100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Friendly Adversarial Training')
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
    parser.add_argument('--grad_clip', type=float, default=5)

    # training config.
    parser.add_argument('--adv_train', type=int, default=1)
    parser.add_argument('--arch_learning_rate', type=float, default=1e-3)
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3)
    parser.add_argument('--use_subset', type=int, default=1)
    parser.add_argument('--is_at', type=int, default=1)
    parser.add_argument('--is_trades', type=int, default=0)
    parser.add_argument('--is_fat', type=int, default=0)
    parser.add_argument('--trades_beta', type=float, default=6.)
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
    parser.add_argument('--graph_hidden_dim', type=int, default=32)
    parser.add_argument("--norm", type=str, default="b")
    parser.add_argument("--init_channel", type=int, default=20)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument('--num_op', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--factor', type=int, default=4)

    # variants.
    parser.add_argument('--is_softmax', type=int, default=0)
    parser.add_argument('--is_fixed', type=int, default=0)
    parser.add_argument('--is_gumbel', type=int, default=0)
    parser.add_argument('--space_version', type=str, default='v1')
    parser.add_argument('--is_normal', type=int, default=0)
    parser.add_argument('--is_quick', type=int, default=1)
    parser.add_argument('--is_coord_descent', type=int, default=0)
    parser.add_argument('--is_score_function', type=int, default=0)
    parser.add_argument('--is_grad_clip', type=int, default=0)
    parser.add_argument('--is_different_optimizers', type=int, default=0)
    parser.add_argument('--start_evaluation', type=int, default=0)
    parser.add_argument('--is_dynamic_temp', type=int, default=0)
    parser.add_argument('--is_diri', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=10)
    parser.add_argument('--use_amp', type=int, default=1)

    # wideresnet variants.
    parser.add_argument('--is_wideresnet', type=int, default=0)
    parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
    parser.add_argument('--wrn_depth', type=int, default=32, help='WRN depth')
    parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')

    args = parser.parse_args()
    print(args)
    import os
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    torch.cuda.set_device(int(args.gpu))
    if args.is_wideresnet:
        if args.wrn_depth == 32:
            model = Wide_ResNet_Madry(
                depth=args.wrn_depth, num_classes=10, 
                widen_factor=args.width_factor, 
                dropRate=args.drop_rate).cuda()
        if args.wrn_depth == 34:
            model = Wide_ResNet(
                depth=args.wrn_depth, 
                num_classes=10, 
                widen_factor=args.width_factor, 
                dropRate=args.drop_rate).cuda()
        # model = torch.nn.DataParallel(model)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # genotype = eval("genotypes.%s" % args.arch)
        model = CIFAR10Module_ARK_Adaptive(args=args,
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


    ############dataloader##########
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
        if args.use_subset:
            testset = torch.utils.data.Subset(testset,
                                            random.sample(range(len(testset)), 2600))
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    # import ipdb; ipdb; ipdb.set_trace()
    print(start_epoch)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # evaluate_epsilon = 
    model.eval()
    if args.is_softmax or args.is_gumbel or args.is_diri:
        model.arch_param.requires_grad = False
    if args.dataset == 'cifar10':
        evaluate_epsilon = 0.031
    else:
        evaluate_epsilon = 0.031
    if args.is_trades:
        step_size = 0.003
    else:
        step_size = 0.031 / 4
    if args.is_gumbel:
        model.set_temperature(args.temperature)
        # print('temp this epoch is:', temp)
    model.arch_param = torch.Tensor([[-3.8046e-02, -4.0694e-02,  1.8031e-01, -5.9434e-02],
        [ 3.0606e-02,  3.3249e-04,  1.5231e-02, -2.3586e-02],
        [ 5.6491e-02, -2.2726e-02, -8.9184e-02,  1.0278e-01],
        [ 5.5741e-04, -2.5018e-02,  4.2670e-02,  3.5855e-02],
        [ 2.9131e-02, -2.3679e-02,  4.5026e-02, -1.9753e-03],
        [ 2.9233e-02, -5.6670e-02, -2.2855e-02,  1.0332e-01],
        [-3.2084e-02, -9.9476e-02,  8.6322e-02,  8.0716e-02],
        [ 3.2483e-02, -7.4212e-02,  6.6345e-02,  4.0340e-02],
        [ 7.5314e-02, -8.5606e-02,  4.5782e-02, -3.2868e-03],
        [ 9.0318e-02, -1.0605e-01,  2.4147e-02,  3.2868e-02],
        [ 1.2734e-01, -8.2794e-02, -1.3688e-01,  1.2983e-01],
        [ 1.3050e-01, -1.1210e-01,  3.1331e-02,  7.6262e-02],
        [ 2.3352e-01, -1.6725e-01,  6.8595e-02, -3.9627e-02],
        [ 1.3590e-01, -1.7111e-01, -2.1120e-02,  1.1846e-01],
        [ 1.1671e+00, -1.2097e+00, -2.7491e-01, -2.9023e-01]]).cuda()






    loss, fgsm_acc = eval_robust(
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
    print(fgsm_acc)
    loss, test_pgd20_acc = eval_robust(
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
    print(test_pgd20_acc)
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
    loss, cw_acc = eval_robust(
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
    print(cw_acc)


    loss, aa_acc = eval_robust_aa(
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
    print(aa_acc)
import argparse
import math
import os
import random
import shutil
import time
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from resnet import *
from utils.utils_data import *
from utils.utils_loss import *
from utils.cub200 import load_cub200

from utils.voc import load_voc

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'sun397', 'voc'],
                    help='dataset name (cifar10)')
# parser.add_argument('--num-class', default=10, type=int,
#                     help='number of class')
parser.add_argument('--exp-dir', default='experiment/cifar10', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--data_dir', default='../codes/data/', type=str,
                    help='experiment directory for loading pre-generated data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 supported)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=800, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800',
                    help='where to decay lrt')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--eta', default=0.9, type=float,
                    help='final weight of reliable sample loss')
parser.add_argument('--t', default=2, type=float,
                    help='tau for logits-adjustment')
parser.add_argument('--alpha_range', default='0.2,0.6', type=str,
                    help='ratio of clean labels (alpha)')
parser.add_argument('--e', default=50, type=int,
                    help='warm-up training')

parser.add_argument('--partial_rate', default=0.3, type=float,
                    help='ambiguity level')
parser.add_argument('--hierarchical', default=False, type=bool,
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--imb_type', default='exp', choices=['exp', 'step'],
                    help='imbalance data type')
parser.add_argument('--imb_ratio', default=100, type=float,
                    help='imbalance ratio for long-tailed dataset generation')
parser.add_argument('--save_ckpt', action='store_true',
                    help='whether save the model')

parser.add_argument('--resume', default='', type=str, help='models path for load')


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Trainer():
    def __init__(self, args):
        self.args = args
        model_path = '{ds}_p{pr}_alpha{alpha}_tau{t}_ep{ep}_e{e}_imb_{it}{imr}_sd_{seed}'.format(
            ds=args.dataset,
            pr=args.partial_rate,
            ep=args.epochs,
            alpha=args.alpha_range,
            it=args.imb_type,
            imr=args.imb_ratio,
            seed=args.seed,
            t=args.t,
            e=args.e,
        )

        args.data_dir_prod = os.path.join(args.data_dir, 'pre-processed-data')
        args.exp_dir = os.path.join(args.exp_dir, model_path)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        if not os.path.exists(args.data_dir_prod):
            os.makedirs(args.data_dir_prod)

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            cudnn.deterministic = True

        if args.dataset == 'cifar10':
            args.num_class = 10
            many_shot_num = 4
            low_shot_num = 3
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt \
                = load_cifar(args=args)

        elif args.dataset == 'cifar100':
            args.num_class = 100
            many_shot_num = 33
            low_shot_num = 33
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt \
                = load_cifar(args=args)
        elif args.dataset == 'sun397':
            input_size = 224
            args.num_class = 397
            many_shot_num = 132
            low_shot_num = 132
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt = load_sun397(
                data_dir=args.data_dir,
                input_size=input_size,
                partial_rate=args.partial_rate,
                batch_size=args.batch_size)
        elif args.dataset == 'voc':
            train_loader, train_givenY, test_loader, train_label_cnt = load_voc(
                batch_size=args.batch_size, con=True)
            args.num_class = 20
            many_shot_num = 6
            low_shot_num = 7
        else:
            raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
        # this train loader is the partial label training loader

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_givenY = train_givenY.cuda()
        # set loss functions (with pseudo-targets maintained)
        self.acc_shot = AccurracyShot(train_label_cnt, args.num_class, many_shot_num, low_shot_num)


    def train(self):
        # create model
        print("=> creating model 'resnet18'")
        if args.dataset in ['sun397', 'voc']:
            print('Loading Pretrained Model')
            model = DHNet_Atten(args.num_class, pretrained=True)
        else:
            model = DHNet_Atten(args.num_class)

        model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # set optimizer
        loss_fn = PLL_loss(self.train_givenY, mu=0.6)
        self.loss_fn = loss_fn

        args.start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                self.loss_fn.confidence = checkpoint['confidence'].cuda()
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))


        best_acc_ens = 0

        for epoch in range(args.start_epoch, args.epochs):
            is_best_ens = False

            adjust_learning_rate(args, optimizer, epoch)

            self.train_loop(model, loss_fn, optimizer, epoch)
            acc_test_tail, acc_many_tail, acc_med_tail, acc_few_tail = self.test(model, self.test_loader, type=1)
            acc_test_head, acc_many_head, acc_med_head, acc_few_head = self.test(model, self.test_loader, type=2)
            acc_test_ens, acc_many_ens, acc_med_ens, acc_few_ens = self.test(model, self.test_loader, type=3)

            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write(
                    'Epoch {}/{}: Acc_tail {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_tail, acc_many_tail, acc_med_tail, acc_few_tail,
                        optimizer.param_groups[0]['lr']))

                f.write(
                    'Epoch {}/{}: Acc_head {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_head, acc_many_head, acc_med_head, acc_few_head,
                        optimizer.param_groups[0]['lr']))
                f.write(
                    'Epoch {}/{}: Acc_ens {:.2f}, Best Acc_ens {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_ens, best_acc_ens, acc_many_ens, acc_med_ens, acc_few_ens,
                        optimizer.param_groups[0]['lr']))

            if acc_test_ens > best_acc_ens:
                best_acc_ens = acc_test_ens
                is_best_ens = True

            if args.save_ckpt:
                self.save_checkpoint({
                    'confidence': loss_fn.confidence.detach(),
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best_ens, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
                    best_file_name='{}/checkpoint_best_ens.pth.tar'.format(args.exp_dir))
            # save checkpoint

    def get_high_confidence(self, loss_vec,  pseudo_label_idx, nums_vec):
        idx_chosen = []
        chosen_flags = torch.zeros(len(loss_vec)).cuda()
        # initialize selection flags
        for j, nums in enumerate(nums_vec):
            indices = np.where(pseudo_label_idx.cpu().numpy() == j)[0]
            # torch.where will cause device error
            if len(indices) == 0:
                continue
                # if no sample is assigned this label1 (by argmax), skip
            loss_vec_j = loss_vec[indices]
            sorted_idx_j = loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(nums)), len(indices)), 1)
            # at least one example
            idx_chosen.append(indices[sorted_idx_j[:partition_j]])

        idx_chosen = np.concatenate(idx_chosen)
        chosen_flags[idx_chosen] = 1

        idx_chosen = torch.where(chosen_flags == 1)[0]
        return idx_chosen

    def get_loss(self, X_w, logits_w, logits_s, ce_label, Y, index, model, loss_fn, emp_dist, alpha, eta, epoch,
                 is_tail):
        bs = X_w.shape[0]
        prediction = F.softmax(logits_w.detach(), dim=1)
        prediction_adj = prediction * Y
        prediction_adj = prediction_adj / prediction_adj.sum(dim=1, keepdim=True)
        # re-normalized prediction for unreliable examples

        _, ce_loss_vec = loss_fn(logits_w, None, targets=ce_label)
        loss_pseu, _ = loss_fn(logits_w, index)

        pseudo_label_idx = ce_label.max(dim=1)[1]
        r_vec = emp_dist * bs * alpha
        idx_chosen = self.get_high_confidence(ce_loss_vec, pseudo_label_idx, r_vec.tolist())

        if epoch < 1 or idx_chosen.shape[0] == 0:
            # first epoch, using uniform labels for training
            #if no samples are chosen
            loss = loss_pseu
        else:
            loss_ce, _ = loss_fn(logits_s[idx_chosen], None, targets=ce_label[idx_chosen])
            # consistency regularization

            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            X_w_c = X_w[idx_chosen]
            ce_label_c = ce_label[idx_chosen]
            idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[idx]
            ce_label_c_rand = ce_label_c[idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
            ce_label_c_mix = l * ce_label_c + (1 - l) * ce_label_c_rand
            if is_tail:
                _, logits_mix, _ = model(X_w_c_mix)
            else:
                logits_mix, _, _ = model(X_w_c_mix)
            loss_mix, _ = loss_fn(logits_mix, None, targets=ce_label_c_mix)
            # mixup training

            loss = (loss_mix + loss_ce) * eta + loss_pseu

        return loss, prediction_adj

    def train_loop(self, model, loss_fn, optimizer, epoch):
        args = self.args
        train_loader = self.train_loader

        batch_time = AverageMeter('Time', ':1.2f')
        data_time = AverageMeter('DataTime', ':1.2f')
        acc_head = AverageMeter('Acc@head', ':2.2f')
        acc_con = AverageMeter('Acc@con', ':2.2f')
        acc_tail = AverageMeter('Acc@tail', ':2.2f')
        acc_en = AverageMeter('Acc@en', ':2.2f')
        loss_head_log = AverageMeter('Loss@head', ':2.2f')
        loss_tail_log = AverageMeter('Loss@tail', ':2.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, acc_head, acc_con, acc_tail, acc_en, loss_head_log, loss_tail_log],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        eta = args.eta * linear_rampup(epoch, args.e)
        alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * linear_rampup(epoch, args.e)
        # calculate weighting parameters

        end = time.time()

        emp_dist_tail = torch.Tensor([1 / args.num_class for _ in range(args.num_class)]).cuda()

        emp_dist_head = loss_fn.confidence.sum(0) / loss_fn.confidence.sum()

        for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # x_s is just used for adding more samples
            X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
            Y_true = true_labels.long().detach().cuda()
            # for showing training accuracy and will not be used when training

            logits_w_head, logits_w_tail, feat_w = model(X_w)
            logits_s_head, logits_s_tail, feat_s = model(X_s)
            pseudo_label = loss_fn.confidence[index]

            loss_head, prediction_head = self.get_loss(X_w, logits_w_head, logits_s_head, pseudo_label, Y, index, model,
                                                     loss_fn, emp_dist_head, alpha, eta, epoch, False)

            logit_adj = F.softmax(logits_w_head - args.t * torch.log(emp_dist_head), dim=1)
            loss_tail, prediction_tail = self.get_loss(X_w, logits_w_tail, logits_s_tail, logit_adj, Y, index, model,
                                                     loss_fn, emp_dist_tail, alpha, eta, epoch, True)

            fusion_pred = model.ensemble(logits_w_head.detach(), logits_w_tail.detach(), emp_dist_head)
            fusion_loss = torch.sum(-pseudo_label * torch.log(fusion_pred+1e-8))/fusion_pred.shape[0]
            ratio = 0.5 * linear_rampup(epoch, args.epochs)
            loss = loss_head + loss_tail + ratio * fusion_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_head_log.update(loss_head.item())
            loss_tail_log.update(loss_tail.item())
            # log accuracy
            acc = accuracy(logits_w_head, Y_true)[0]
            acc_head.update(acc[0])

            acc = accuracy(logits_w_tail, Y_true)[0]
            acc_tail.update(acc[0])

            acc = accuracy(fusion_pred.detach(), Y_true)[0]
            acc_en.update(acc[0])

            acc = accuracy(pseudo_label, Y_true)[0]
            acc_con.update(acc[0])

            loss_fn.confidence_move_update(prediction_tail, index)
            # update confidences

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

    def test(self, model, test_loader, type=1):
        with torch.no_grad():
            if type == 1:
                print('==> Evaluation tail...')
            elif type == 2:
                print('==> Evaluation head...')
            else:
                print('==> Evaluation ensemble...')
            model.eval()
            pred_list = []
            true_list = []
            for _, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                if type == 1:
                    _, outputs, _ = model(images)
                    pred = F.softmax(outputs, dim=1)
                elif type == 2:
                    outputs, _, _ = model(images)
                    pred = F.softmax(outputs, dim=1)
                else:
                    logit_head, logit_tail, _ = model(images)
                    pred = model.ensemble(logit_head, logit_tail, self.loss_fn.get_distribution())

                pred_list.append(pred.cpu())
                true_list.append(labels)

            pred_list = torch.cat(pred_list, dim=0)
            true_list = torch.cat(true_list, dim=0)

            acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
            acc_many, acc_med, acc_few = self.acc_shot.get_shot_acc(pred_list.max(dim=1)[1], true_list)
            print('==> Test Accuracy is %.2f%% (%.2f%%), [%.2f%%, %.2f%%, %.2f%%]' % (
                acc1, acc5, acc_many, acc_med, acc_few))
        return float(acc1), float(acc_many), float(acc_med), float(acc_few)


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_file_name)


if __name__ == '__main__':
    args = parser.parse_args()

    [args.alpha_start, args.alpha_end] = [float(item) for item in args.alpha_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.imb_factor = 1. / args.imb_ratio
    print(args)

    # set imb_factor as 1/imb_ratio
    trainer = Trainer(args)
    trainer.train()

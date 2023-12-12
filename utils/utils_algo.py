import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


def sinkhorn(pred, eta, r_in=None, rec=False):
    PS = pred.detach()
    K = PS.shape[1]
    N = PS.shape[0]
    PS = PS.T
    c = torch.ones((N, 1)) / N
    r = r_in.cuda()
    c = c.cuda()
    # average column mean 1/N
    PS = torch.pow(PS, eta)  # K x N
    r_init = copy.deepcopy(r)
    inv_N = 1. / N
    err = 1e6
    # error rate
    _counter = 1
    for i in range(50):
        if err < 1e-1:
            break
        r = r_init * (1 / (PS @ c))  # (KxN)@(N,1) = K x 1
        # 1/K(Plambda * beta)
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        # 1/N(alpha * Plambda)
        if _counter % 10 == 0:
            err = torch.sum(c_new) + torch.sum(r)
            if torch.isnan(err):
                # This may very rarely occur (maybe 1 in 1k epochs)
                # So we do not terminate it, but return a relaxed solution
                print('====> Nan detected, return relaxed solution')
                pred_new = pred + 1e-5 * (pred == 0)
                relaxed_PS, _ = sinkhorn(pred_new, eta, r_in=r_in, rec=True)
                z = (1.0 * (pred != 0))
                relaxed_PS = relaxed_PS * z
                return relaxed_PS, True
        c = c_new
        _counter += 1
    PS *= torch.squeeze(c)
    PS = PS.T
    PS *= torch.squeeze(r)
    PS *= N
    return PS.detach(), False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr

    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if steps > 0:
        lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''
    augmentation for real world dataset
    x: feature n*c
    rate: 覆盖的比例
    return n*c
'''
def augmentation_for_feature(x, rate):
    mask_num = int(x.shape[1] * rate)
    p = torch.ones_like(x)
    mask_indexes = torch.multinomial(p, mask_num, False)
    return torch.scatter(x, dim=1, index=mask_indexes, value=0)


class AccurracyShot(object):
    def __init__(self, train_class_count, num_class, many_shot_num=3, low_shot_num=3):
        self.train_class_count = train_class_count
        self.test_class_count = None
        self.num_class = num_class
        self.many_shot_thr = train_class_count.sort()[0][num_class - many_shot_num - 1]
        self.low_shot_thr = train_class_count.sort()[0][low_shot_num]

    def get_shot_type_index(self, labels):
        many_idx = []
        medium_idx = []
        few_idx = []
        for i in range(self.num_class):
            idx_i = torch.where(labels == i)[0].tolist()
            if self.train_class_count[i] > self.many_shot_thr:
                many_idx.extend(idx_i)
            elif self.train_class_count[i] < self.low_shot_thr:
                medium_idx.extend(idx_i)
            else:
                few_idx.extend(idx_i)
        return many_idx, medium_idx, few_idx

    def get_shot_acc(self, preds, labels, acc_per_cls=False):
        if self.test_class_count is None:
            self.test_class_count = []
            for l in range(self.num_class):
                self.test_class_count.append(len(labels[labels == l]))

        class_correct = []
        for l in range(self.num_class):
            class_correct.append((preds[labels == l] == labels[labels == l]).sum())

        many_shot = []
        median_shot = []
        low_shot = []
        for i in range(self.num_class):
            if self.train_class_count[i] > self.many_shot_thr:
                many_shot.append((class_correct[i] / float(self.test_class_count[i])))
            elif self.train_class_count[i] < self.low_shot_thr:
                low_shot.append((class_correct[i] / float(self.test_class_count[i])))
            else:
                median_shot.append((class_correct[i] / float(self.test_class_count[i])))

        if len(many_shot) == 0:
            many_shot.append(0)
        if len(median_shot) == 0:
            median_shot.append(0)
        if len(low_shot) == 0:
            low_shot.append(0)

        if acc_per_cls:
            class_accs = [c / cnt for c, cnt in zip(class_correct, self.test_class_count)] 
            return np.mean(many_shot) * 100, np.mean(median_shot) * 100, np.mean(low_shot) * 100, class_accs
        else:
            return np.mean(many_shot) * 100, np.mean(median_shot) * 100, np.mean(low_shot) * 100

def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        if len(target.shape) == 2:
            target = torch.argmax(target, dim=1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


from sklearn.metrics import average_precision_score

def calculate_mAP(y_true, y_scores):
    """
    Calculate the mean average precision (mAP) for multi-class classification tasks.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    num_class = y_scores.shape[1]
    APs = []
    for i in range(num_class):
        y_true_i = np.copy(y_true)
        y_true_i[y_true != i] = 0
        y_true_i[y_true == i] = 1
        y_scores_i = y_scores[:, i]
        ap = average_precision_score(y_true_i, y_scores_i)
        APs.append(ap)
    return np.mean(APs)

def test_for_each_class(model, test_loader, distribution):
    with torch.no_grad():
        model.eval()
        pred_list = []
        true_list = []
        for _, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            logit_imb, logit_bal, _ = model(images)
            pred = model.ensemble(logit_imb, logit_bal, distribution)

            pred_list.append(pred.cpu())
            true_list.append(labels)

        pred_list = torch.cat(pred_list, dim=0)
        true_list = torch.cat(true_list, dim=0)
        pred_labels = pred_list.max(1)[1]
        acc_all = []
        for i in range(int(true_list.max())+1):
            p = pred_labels[true_list == i]
            acc_all.append(float(torch.sum(p == i)/len(p)))

        acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
        print('==> Test Accuracy is %.2f%% (%.2f%%)' % (acc1, acc5))
        print(acc_all)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    # ax = plt.subplot(111)       # 创建子图

    # 遍历所有样本
    for i in range(max(label) + 1):
        # 在图中为每个数据点画出标签
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
        #          fontdict={'weight': 'bold', 'size': 7})
        l = label == i
        x, y = data[l].T
        plt.scatter(x, y, label=str(i))
    plt.legend()
    # plt.xticks()        # 指定坐标的刻度
    # plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig


def show_embedding(data, label, title):
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    reslut = ts.fit_transform(data)
    # 调用函数，绘制图像
    fig = plot_embedding(reslut, label, title)
    # 显示图像
    plt.show()


def generate_weight(n_classes, n_hiddens, epochs=20000,  use_relu=False):
    n_samples = n_classes
    labels = torch.arange(0, n_samples).cuda()
    scale = 5
    Z = torch.randn(n_samples, n_hiddens).cuda()
    Z.requires_grad = True
    W = torch.randn(n_classes, n_hiddens).cuda()
    W.requires_grad = True
    nn.init.kaiming_normal_(W)
    optimizer = SGD([Z, W], lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20000, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    for i in range(epochs):
        if use_relu:
            z = F.relu(Z)
        else:
            z = Z
        w = W
        L2_z = F.normalize(z, dim=1)
        L2_w = F.normalize(w, dim=1)
        out = F.linear(L2_z, L2_w)
        loss = criterion(out * scale, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return W

if __name__ == '__main__':
    W = generate_weight(10, 512, 20000)

    print(W)
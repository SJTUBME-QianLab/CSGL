import torch
import numpy as np
import pandas as pd
import math
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore")


class IOStream:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


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


def adjust_learning_rate(epoch, args, optimizer):
    # steps = np.sum(epoch > np.asarray(args.lr_decay))
    steps = np.sum(epoch > np.asarray(args.mile_stones))
    if steps > 0:
        new_lr = args.lr * (args.lr_decay ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def adjust_mask_ratio(epoch, args):
    """ log step Mask ratio """
    io = args.io
    if epoch < args.pre_epochs:
        io.cprint('Pretrain Epoch {}, lr{}, mr{}'.format(epoch, args.lr, args.mr))
        args.mr0 = args.mr
    elif (epoch < args.mr_mile_stones[-1]) & (args.fs_times > 0):
        mr_decay_times = (epoch - args.pre_epochs) // args.epoch_per_fs + 1
        mr_values = np.logspace(start=np.log(args.mr0) / np.log(10),
                                stop=np.log(args.fea_num / args.data_dim) / np.log(10),
                                num=args.fs_times + 1, endpoint=True)
        args.mr = mr_values[mr_decay_times]
        io.cprint('Fea. Select Epoch {}, Mask Ratio {:2.3f}, lr{}:'.format(epoch, args.mr, args.lr))
    else:
        args.mr = args.fea_num / args.data_dim
        io.cprint('Final Tune Epoch {}, Mask Ratio {:2.3f}, lr{}:'.format(epoch, args.mr, args.lr))


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def set_optimizer(model, args):
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(params=model.emb_head.parameters(),
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(params=model.emb_head.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)


    return optimizer


def save_model(save_file, epoch, model, optimizer, args,
               loss_pos, loss_neg, loss_clf, acc_clf, loss_val, acc_val):
    print('==> Saving...')
    state = dict(
        epoch=epoch,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        mr=args.mr,
        loss_pos=loss_pos,
        loss_neg=loss_neg,
        loss_clf_tr=loss_clf,
        acc_clf_tr=acc_clf,
        loss_clf_te=loss_clf,
        acc_clf_te=acc_clf,
        loss_val=loss_val,
        acc_val=acc_val
    )
    if args.encoder == 'Att':
        state['model_att'] = model.atten_scores,
        state['model_fea'] = model.fea_scores,

    torch.save(state, save_file)


def evaluate(prob, true_onehot=None, true=None):
    # calculate
    pred = np.argmax(prob, axis=1)
    if true_onehot is None and true is None:
        raise ValueError
    if true_onehot is None:
        true_onehot = pd.get_dummies(true).values
    if true is None:
        true = np.argmax(true_onehot, axis=1)
    acc = sklearn.metrics.accuracy_score(true, pred)
    recall = sklearn.metrics.recall_score(true, pred, average='macro')
    precision = sklearn.metrics.precision_score(true, pred, average='macro')
    f1score = sklearn.metrics.f1_score(true, pred, average='macro')

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true, pred).ravel()
    # print([tn, fp, fn, tp])
    fpr, tpr, _ = sklearn.metrics.roc_curve(true_onehot.ravel(), prob.ravel())
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    specificity = tn / (tn + fp)

    # roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    # roc.to_csv(fold_dir + '/final_roc.csv')

    return [acc, roc_auc, recall, precision, f1score, specificity], fpr, tpr


def evaluate_pred(pred, true):
    acc = sklearn.metrics.accuracy_score(true, pred)
    recall = sklearn.metrics.recall_score(true, pred, average='macro')
    precision = sklearn.metrics.precision_score(true, pred, average='macro')
    f1score = sklearn.metrics.f1_score(true, pred, average='macro')
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true, pred).ravel()
    specificity = tn / (tn + fp)

    return [acc, recall, precision, f1score, specificity]


def onehot_01_code(Y):
    Yc_onehot = np.zeros((len(Y), 2))
    Yc_onehot[np.where(Y == 1)[0], 1] = 1.0
    Yc_onehot[np.where(Y == 0)[0], 0] = 1.0

    return Yc_onehot


def concat_list(fea_scores_list):
    if isinstance(fea_scores_list[0], list):
        opt_fea_scores = [torch.zeros(fea_scores_list[0][i].shape).to(fea_scores_list[0][i].device)
                          for i in range(len(fea_scores_list[0]))]
        for fea_scores in fea_scores_list:  # fea_scores: list12
            for g, score in enumerate(fea_scores):
                opt_fea_scores[g] += score / len(fea_scores_list)
        att = torch.cat(opt_fea_scores, dim=0).cpu().numpy()
    else:
        opt_fea_scores = torch.mean(torch.cat(fea_scores_list, dim=1), dim=1)
        att = opt_fea_scores.cpu().numpy()

    return opt_fea_scores, att


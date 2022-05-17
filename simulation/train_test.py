import numpy as np
import os
import time
from tools.utils import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


class TrainTestConNegMask:
    def __init__(self, logger):
        self.obj = torch.nn.MSELoss(reduction='mean')  # *1/n
        self.CE = torch.nn.CrossEntropyLoss()
        self.sft = torch.nn.Softmax(dim=1)
        self.logger = logger

    def train(self, epoch, train_loader, model, optimizer, opt_beta, args):
        losses_all = AverageMeter('Train_all_loss', ':.4f')
        losses_obj = AverageMeter('Train_obj_loss', ':.4f')
        losses_L1 = AverageMeter('Train_L1_loss', ':.4f')
        losses_L21 = AverageMeter('Train_L21_loss', ':.4f')
        losses_Lg = AverageMeter('Train_Lg_loss', ':.4f')
        losses_neg = AverageMeter('Train_neg_loss', ':.4f')
        sparsities = AverageMeter('Sparsity', ':.4f')
        acces = AverageMeter('Train_Acc', ':2.2f')

        io = args.io
        model.train()
        pred_, true_ = [], []

        for i, (X, Y, pair) in enumerate(train_loader):

            if X.shape[1] == (model.dim_in - 1):
                add_ones = torch.ones((X.shape[0], 1)).to(X.device)
                # X = torch.concat([X, add_ones], dim=1)
                X = torch.cat([X, add_ones], dim=1)
            bsz = len(Y)

            sparsity = model.beta.data.count_nonzero() / model.beta.data.numel()
            sparsities.update(sparsity.item())

            pred = model(X, obj=True)
            # pred = torch.where(out > 0, 1, 0)
            pred_.append(pred.data.cpu().numpy())
            true_.append(Y.data.cpu().numpy())
            acc = torch.where(pred > 0, 1, 0).view_as(Y).eq(Y).sum() / Y.numel()
            acces.update(acc.item())

            # objective loss
            loss_obj = self.obj(Y.to(pred.dtype).view_as(pred), pred)
            losses_obj.update(loss_obj.item(), bsz)

            loss_all = loss_obj.clone()

            # Update Contrastive Module (model.emb_head)
            if args.Lcon > 0:
                loss_neg = model.get_Lcon_neg_mask(X, pair)
                losses_neg.update(loss_neg.item(), bsz)
                loss_all += args.Lcon * loss_neg

                if epoch % args.jump == 0:
                    optimizer.zero_grad()
                    loss_neg.backward()
                    optimizer.step()

            # Update penalized SGL Module (model.beta)
            if args.L1 > 0:
                L1 = model.get_L1()
                losses_L1.update(L1.item(), bsz)
                loss_all += args.L1 * L1
            if args.L21 > 0:
                L21 = model.get_L21()
                losses_L21.update(L21.item(), bsz)
                loss_all += args.L21 * L21
            if args.Lg > 0:
                Lg = model.get_Lg()
                losses_Lg.update(Lg.item(), bsz)
                loss_all += args.Lg * Lg
            losses_all.update(loss_all, bsz)

            opt_beta.update_beta(X, Y, pair, t0_step=args.t0_step)

            report_str = 'Train: [{}][{}/{}], sparsity {:2.2f}' \
                         '\tloss_all {loss_all.val:.3f} ({loss_all.avg:.3f})' \
                         '\tloss_obj {loss_obj.val:.3f} ({loss_obj.avg:.3f})' \
                         '\tacc {acc.val:.3f} ({acc.avg:.3f})'. \
                         format(epoch, i + 1, len(train_loader), sparsity.item(),
                                loss_all=losses_all, loss_obj=losses_obj, acc=acces)
            if args.L1 > 0:
                report_str += '\tloss_L1 {loss_L1.val:.3f} ({loss_L1.avg:.3f})'.format(loss_L1=losses_L1)
            if args.L21 > 0:
                report_str += '\tloss_L21 {loss_L21.val:.3f} ({loss_L21.avg:.3f})'.format(loss_L21=losses_L21)
            if args.Lg > 0:
                report_str += '\tloss_Lg {loss_Lg.val:.3f} ({loss_Lg.avg:.3f})'.format(loss_Lg=losses_Lg)
            if args.Lcon > 0:
                report_str += '\tloss_neg {loss_neg.val:.3f} ({loss_neg.avg:.3f})'.format(loss_neg=losses_neg)
            io.cprint(report_str)

        pred_ = np.concatenate(pred_, axis=0)
        true_ = np.concatenate(true_, axis=0)

        losses_list = [losses_all.avg, losses_obj.avg, losses_L1.avg, losses_L21.avg, losses_Lg.avg, losses_neg.avg]

        self.logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        self.logger.log_value('Train_all', losses_all.avg, epoch)
        self.logger.log_value('Train_obj', losses_obj.avg, epoch)
        self.logger.log_value('Train_L1', losses_L1.avg, epoch)
        self.logger.log_value('Train_L21', losses_L21.avg, epoch)
        self.logger.log_value('Train_Lg', losses_Lg.avg, epoch)
        self.logger.log_value('Train_neg', losses_neg.avg, epoch)
        self.logger.log_value('Sparsity', sparsities.avg, epoch)
        self.logger.log_value('Train_Acc', acces.avg, epoch)

        return losses_list, acces.avg, pred_, true_, model.beta[:-1].detach().cpu().numpy()

    def test(self, epoch, test_loader, model, args):
        losses_all = AverageMeter('Test_all_loss', ':.4f')
        losses_obj = AverageMeter('Test_obj_loss', ':.4f')
        losses_L1 = AverageMeter('Test_L1_loss', ':.4f')
        losses_L21 = AverageMeter('Test_L21_loss', ':.4f')
        losses_Lg = AverageMeter('Test_Lg_loss', ':.4f')
        losses_neg = AverageMeter('Test_neg_loss', ':.4f')
        acces = AverageMeter('Test_Acc', ':2.2f')

        io = args.io
        model.eval()
        pred_, true_ = [], []

        for i, (X, Y, pair) in enumerate(test_loader):
            if X.shape[1] == (model.dim_in - 1):
                add_ones = torch.ones((X.shape[0], 1)).to(X.device)
                # X = torch.concat([X, add_ones], dim=1)
                X = torch.cat([X, add_ones], dim=1)
            bsz = len(Y)

            pred = model(X, obj=True)
            # pred = torch.where(out > 0, 1, 0)
            pred_.append(pred.data.cpu().numpy())
            true_.append(Y.data.cpu().numpy())
            acc = torch.where(pred > 0, 1, 0).view_as(Y).eq(Y).sum() / Y.numel()
            acces.update(acc.item())

            # objective loss
            loss_obj = self.obj(Y.to(pred.dtype).view_as(pred), pred)
            losses_obj.update(loss_obj.item(), bsz)

            loss_all = loss_obj.clone()

            # Update Contrastive Module (model.emb_head)
            if args.Lcon > 0:
                loss_neg = model.get_Lcon_neg_mask(X, pair)
                losses_neg.update(loss_neg.item(), bsz)
                loss_all += args.Lcon * loss_neg

            # Update penalized SGL Module (model.beta)
            if args.L1 > 0:
                L1 = model.get_L1()
                losses_L1.update(L1.item(), bsz)
                loss_all += args.L1 * L1
            if args.L21 > 0:
                L21 = model.get_L21()
                losses_L21.update(L21.item(), bsz)
                loss_all += args.L21 * L21
            if args.Lg > 0:
                Lg = model.get_Lg()
                losses_Lg.update(Lg.item(), bsz)
                loss_all += args.Lg * Lg
            losses_all.update(loss_all, bsz)

            report_str = 'Test: [{0}][{1}/{2}]' \
                         '\tloss_all {loss_all.val:.3f} ({loss_all.avg:.3f})' \
                         '\tloss_obj {loss_obj.val:.3f} ({loss_obj.avg:.3f})' \
                         '\tacc {acc.val:.3f} ({acc.avg:.3f})'. \
                         format(epoch, i + 1, len(test_loader),
                                loss_all=losses_all, loss_obj=losses_obj, acc=acces)
            if args.L1 > 0:
                report_str += '\tloss_L1 {loss_L1.val:.3f} ({loss_L1.avg:.3f})'.format(loss_L1=losses_L1)
            if args.L21 > 0:
                report_str += '\tloss_L21 {loss_L21.val:.3f} ({loss_L21.avg:.3f})'.format(loss_L21=losses_L21)
            if args.Lg > 0:
                report_str += '\tloss_Lg {loss_Lg.val:.3f} ({loss_Lg.avg:.3f})'.format(loss_Lg=losses_Lg)
            if args.Lcon > 0:
                report_str += '\tloss_neg {loss_neg.val:.3f} ({loss_neg.avg:.3f})'.format(loss_neg=losses_neg)
            io.cprint(report_str)

        pred_ = np.concatenate(pred_, axis=0)
        true_ = np.concatenate(true_, axis=0)

        losses_list = [losses_all.avg, losses_obj.avg, losses_L1.avg, losses_L21.avg, losses_Lg.avg, losses_neg.avg]

        self.logger.log_value('Test_all', losses_all.avg, epoch)
        self.logger.log_value('Test_obj', losses_obj.avg, epoch)
        self.logger.log_value('Test_L1', losses_L1.avg, epoch)
        self.logger.log_value('Test_L21', losses_L21.avg, epoch)
        self.logger.log_value('Test_Lg', losses_Lg.avg, epoch)
        self.logger.log_value('Test_neg', losses_neg.avg, epoch)
        self.logger.log_value('Test_Acc', acces.avg, epoch)

        return losses_list, acces.avg, pred_, true_


def eval_6clf(idx, fea_name, fea_ind, loc, ch_label, data):
    sheet_name = ['svm', 'lr', 'nb', 'knn', 'dt', 'rf']
    eval_name = ['acc', 'recall', 'precision', 'f1score', 'specificity']
    all_eval = np.zeros((len(sheet_name), len(idx), len(eval_name)))
    [X_train, Y_train, X_test, Y_test] = data

    for i in range(len(idx)):
        idx_tmp = idx[:(i + 1)]
        X_tr_tmp = X_train[:, idx_tmp]
        X_te_tmp = X_test[:, idx_tmp]

        # svm
        j = 0
        svm_clf = Pipeline([("scaler", StandardScaler()), ("svc_rbf", SVC(kernel="rbf"))])
        svm_clf.fit(X_tr_tmp, Y_train)
        pred = svm_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # lr
        j += 1
        oe = OrdinalEncoder()
        # oe.fit(Y_train.reshape(-1, 1)).categories_
        Y_train0 = oe.fit_transform(Y_train.reshape(-1, 1))
        lr_clf = LogisticRegression()
        lr_clf.fit(X_tr_tmp, Y_train0.reshape(-1))
        pred = lr_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # nb
        j += 1
        nb_clf = GaussianNB()
        nb_clf.fit(X_tr_tmp, Y_train)
        pred = nb_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # knn
        j += 1
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_tr_tmp, Y_train)
        pred = knn_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # dt
        j += 1
        dt_clf = DecisionTreeRegressor(random_state=2022)
        dt_clf.fit(X_tr_tmp, Y_train)
        pred = dt_clf.predict(X_te_tmp).astype(int)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

        # rf
        j += 1
        rf_clf = RandomForestClassifier(random_state=2022)
        rf_clf.fit(X_tr_tmp, Y_train)
        pred = rf_clf.predict(X_te_tmp)
        all_eval[j, i, :] = evaluate_pred(pred, Y_test)

    # basic info
    info = pd.DataFrame({
        'NO': range(1, len(fea_name) + 1),
        'fea_ind': fea_ind,
        'fea_name': fea_name,
        'ch': ch_label,
        'loc': loc,
    })
    info_slc = info.iloc[idx, :].reset_index(drop=True, inplace=False)
    info_slc['NO'] = range(1, len(info_slc) + 1)

    df_all = {}
    for i in range(len(all_eval)):
        df = pd.concat([info_slc, pd.DataFrame(all_eval[i], columns=eval_name)], axis=1)
        df_all[sheet_name[i]] = df

    return df_all, info


def eval_FS(fea_info, individuals, sco, result_dir):
    final_weight, metric_i = performance(sco, fea_info, individuals)
    df_groupby1, df_groupby2, df_true_std = group_check(fea_info, final_weight)

    # pd_writer = pd.ExcelWriter(os.path.join(result_dir, 'final_%d_eval0.xlsx' % top_num))
    # acc_TF = []
    # for i, df in df_all_eval.items():
    #     df.to_excel(pd_writer, index=False, index_label=True, sheet_name=i)
    #     acc_TF.append(df.loc[len(df) - 1, 'acc'])
    # pd_writer.save()
    # acc_TF = pd.DataFrame([acc_TF], columns=list(df_all_eval.keys()))

    pd_writer = pd.ExcelWriter(os.path.join(result_dir, 'eval_FS.xlsx'))
    # acc_TF.to_excel(pd_writer, index=False, index_label=True, sheet_name="acc_allT")
    final_weight.to_excel(pd_writer, index=False, index_label=True, sheet_name="final_weight")
    metric_i.to_excel(pd_writer, index=False, index_label=True, sheet_name="metrices")
    df_groupby1.to_excel(pd_writer, index=True, index_label=True, sheet_name="group_check1")
    df_groupby2.to_excel(pd_writer, index=True, index_label=True, sheet_name="group_check2")
    df_true_std.to_excel(pd_writer, index=True, index_label=True, sheet_name="df_true_std")
    pd_writer.save()


def get_data(data_dir, data_name):
    # Xall = np.load(os.path.join(data_dir, data_name, 'X_normL2.npy'))
    # Yall = np.load(os.path.join(data_dir, data_name, 'Y.npy'))
    #
    # test_idx = np.loadtxt(os.path.join(data_dir, data_name, 'test_idx', '%d.txt' % fold)).astype(int)
    # train_idx = np.array(list(set(np.arange(len(Yall))) - set(test_idx)), dtype=int)
    #
    # # 数据
    # data = [Xall, Yall]

    # 基本信息
    fea_info = pd.read_csv(os.path.join(data_dir, data_name, 'basic_info.csv')).iloc[:, 1:]
    # fea_name, gp_label, loc, true_01, isol_01, beta
    # 个性化的特征
    individuals = pd.read_csv(os.path.join(data_dir, data_name, 'spacial_mask.csv'), header=None)

    return fea_info, individuals


def performance(sco, fea_info, individuals):
    fea_name = fea_info['fea_name'].values
    fea_idxes = fea_info.index.values
    locs = fea_info['loc'].values
    gp_info = fea_info['gp_label'].values
    true_beta = fea_info['beta'].values
    TorF = fea_info['true_01'].values

    sco.columns = ['index', 'final_weight']
    sco.insert(loc=2, column='abs_weight', value=abs(sco['final_weight']))
    sco_rank = sco.sort_values(['abs_weight'], ascending=False, kind='mergesort').reset_index(drop=True)

    # slc_idx = np.hstack([sco_rank['index'][:top_num], sco_rank['index'][int(np.sum(TorF))]])

    # basic info
    info = pd.DataFrame({
        'NO': range(1, len(fea_name) + 1),
        'fea_ind': fea_idxes,
        'fea_name': fea_name,
        'ch': gp_info,
        'loc': locs,
    })

    info = pd.merge(info, sco, left_on='fea_ind', right_on='index').drop(['index'], axis=1, inplace=False)

    # Feature selection performance
    true_beta_df = pd.DataFrame({'true_beta': true_beta})
    # TorF = (abs(beta) > 0).astype(float)
    true_beta_df.insert(1, 'TorF', TorF)
    final_weight = pd.concat([info, true_beta_df], axis=1)
    final_weight.insert(loc=list(final_weight.columns).index('abs_weight') + 1,
                        column='abs_weight_normalize',
                        value=final_weight[['abs_weight']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)),
                                                                 axis=0))

    pred_top = np.zeros((1, len(TorF)))
    slc_idx = sco_rank['index'][:int(np.sum(TorF))]
    pred_top[0, slc_idx] = 1
    prob_1 = final_weight['abs_weight_normalize']
    prob = np.vstack([1 - prob_1, prob_1]).T
    metric_i, fpr, tpr = evaluate(prob, true=final_weight['TorF'].values)

    TPR = np.sum((pred_top * TorF)) / np.sum(TorF)
    sparsity = np.sum(final_weight['abs_weight'] > 0) / len(final_weight)

    indi_ft_idx = np.where(np.sum(individuals, axis=1) > 0)[0]
    indi_weight = final_weight.loc[indi_ft_idx, 'abs_weight_normalize'].mean()

    metric_i = pd.DataFrame(np.array([[TPR, sparsity, indi_weight] + metric_i]),
                            columns=['top-TPR', 'sparsity', 'indi_abs_w_mean',
                                     'acc', 'roc_auc', 'recall', 'precision', 'f1score', 'specificity'])

    return final_weight, metric_i


def group_check(fea_info, final_weight):
    df = fea_info.reset_index(drop=False, inplace=False)
    df.rename(columns={df.columns[0]: 'index'}, inplace=True)
    df[['index']] = (df[['index']] + 1).astype(int)
    df.loc[np.where(df[['isol_01']] == 1)[0], 'true_01'] = 2
    df.drop(['index', 'fea_name', 'loc', 'beta'], axis=1, inplace=True)

    df = pd.concat([df, final_weight[['abs_weight', 'abs_weight_normalize']]], axis=1)

    df_groupby1 = df.groupby(['true_01', 'gp_label']).mean()
    df_groupby2 = df.groupby(['gp_label', 'true_01']).mean()
    df_true_std = df[df['true_01'] == 1].groupby(['gp_label']).std()

    return df_groupby1, df_groupby2, df_true_std


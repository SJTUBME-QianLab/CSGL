"""
Sparse Group Lasso model with Contrastive Head
"""
import torch
import torch.nn as nn
import numpy as np
from numpy.linalg.linalg import norm as p_norm
from models.contrastive_simple import SupConLossPairNegMask


class ContrastiveSGL(nn.Module):
    def __init__(self, dim_in, dim_emb, device, gp_idx_list, init, data,
                 temp, fix=True, threshold=1e-3, prune=1.0):
        super(ContrastiveSGL, self).__init__()

        self.dim_in = dim_in + 1  # +1是为了偏置项
        self.dim_emb = dim_emb
        self.device = device
        self.gp_idx_list = gp_idx_list
        self.init = init
        self.fix = fix
        if fix:
            self.threshold = threshold
            self.prune = None
        else:
            self.threshold = None
            self.prune = prune

        # SGL (Sparse Group Lasso) Module
        if self.init == 'randn':
            self.beta = nn.Parameter(torch.randn(self.dim_in, 1), requires_grad=True)
        elif self.init == 'ridge':
            from sklearn import linear_model
            X, y = data
            clf = linear_model.Ridge(alpha=0.1)
            clf.fit(X.cpu().numpy(), y.cpu().numpy())
            self.beta = nn.Parameter(
                torch.from_numpy(np.hstack([clf.coef_, clf.intercept_]).reshape(-1, 1)).float(), requires_grad=True)

        # Graph Penalty Module
        # self.L_list = self.calculate_graph(locs, gp_idx_list, Gkind)

        # Contrastive Module
        self.emb_head = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_emb),
            nn.ReLU(inplace=True)
        )
        self.criterion_con = SupConLossPairNegMask(temperature=temp).to(device)

    def calculate_graph_X(self, X, locs, gp_idx_list, Gkind, gamma=None):
        # Graph Penalty Module
        if Gkind == '':
            L_list = [None for i in gp_idx_list]
        elif Gkind == 'Gau' or Gkind == 'one':
            L_list = self.calculate_graph(locs, gp_idx_list, Gkind)
        elif Gkind == 'Gaux':
            L0_list = self.calculate_graph(locs, gp_idx_list, Gkind[:3])
            X_corr_list = self.X_corr(X, gp_idx_list)
            L_list = [torch.mul(L_g, covX_g) for L_g, covX_g in zip(L0_list, X_corr_list)]
        elif Gkind == 'Gau+':
            L0_list = self.calculate_graph(locs, gp_idx_list, Gkind[:3])
            X_corr_list = self.X_corr(X, gp_idx_list)
            assert 0 <= gamma < 1
            L_list = [gamma * L_g + (1-gamma) * covX_g for L_g, covX_g in zip(L0_list, X_corr_list)]
        else:
            raise ValueError(Gkind)
        self.L_list = L_list

    def X_corr(self, X, gp_idx_list):
        X_corr_list = [np.corrcoef(X[:, gp_idx].T.cpu().numpy()) for gp_idx in gp_idx_list]
        X_corr_list = [torch.from_numpy(G).float().to(self.device) for G in X_corr_list]
        return X_corr_list

    def calculate_graph(self, locs, gp_idx_list, Gkind):
        if Gkind == '':
            L_list = [None for i in gp_idx_list]
        else:
            if Gkind == 'Gau':
                G_list = [Gaussian_matrix(locs[gp_idx]) for gp_idx in gp_idx_list]
            elif Gkind == 'one':
                inter_dim_list = [len(gp_idx) for gp_idx in gp_idx_list]
                G_list = [np.ones((x, x)) / x for x in inter_dim_list]
            else:
                raise ValueError(Gkind)
            G_list = [torch.from_numpy(G).float().to(self.device) for G in G_list]
            L_list = [G.sum(dim=0).diag() - G for G in G_list]

        return L_list

    # For inference only
    def forward(self, x, obj):
        if obj:
            out = x.mm(self.beta)
            return out
        else:
            fea_in = x.mm(self.beta.view(-1).diag())
            fea_out = self.emb_head(fea_in)
            return fea_out

    def get_L1(self):
        # return self.beta.data.clone().detach().abs().sum()
        return torch.norm(self.beta.detach(), p=1)

    def get_L21(self):
        out = 0.
        for gp_idx in self.gp_idx_list:
            out += torch.norm(self.beta[gp_idx].detach(), p=2) * (len(gp_idx) ** 0.5)
            # out += torch.norm(self.beta[gp_idx].detach(), p=2)
        return out

    def get_Lg(self):
        out = 0.
        for L, gp_idx in zip(self.L_list, self.gp_idx_list):
            betai = self.beta[gp_idx].detach()
            # L_norm = L.norm()
            L_norm = 1.
            out += betai.t().mm(L).mm(betai) * L_norm
        return out[0][0]

    # def get_Lcon_neg(self, X, Y, pair):
    #     emb = self.forward(X, obj=False)
    #     loss_neg = self.criterion_con(emb, Y, pair)
    #     return -loss_neg  # ！！！！！！！！！！！！！！argmax

    def get_Lcon_neg_mask(self, X, pair):
        emb = self.forward(X, obj=False)
        loss_neg = self.criterion_con(emb, pair)
        return -loss_neg  # ！！！！！！！！！！！！！！argmax

    # def get_Lcon_pos(self, X, Y):
    #     emb = self.forward(X, obj=False)
    #     loss_pos = self.criterion_con(emb, Y)
    #     return loss_pos


class UpdateBeta:
    def __init__(self, model, args):
        self.model = model
        self.lam1 = args.L1
        self.lam2 = args.L21
        self.lam_Gau = args.Lg
        self.lam_con = args.Lcon
        self.io = args.io

    """
    def update_beta(self, X, Y, pair, t0=1, t0_step=1.5):
        beta_prev = self.model.beta.data.clone()
        grad_con = self.get_Contrastive_grad(X, pair)

        stop_cond = False
        t = t0 / 1000
        i = 1
        while not stop_cond:
            self.update_beta_once(beta_prev, X, Y, pair, t)

            if self.lam_con > 0:
                stop_cond = self.validate_majorization(t, X, pair, grad_con, beta_prev)
                t *= t0_step  # enlarge step-size
                if stop_cond:
                    self.io.cprint('Majorization succeeded (%d).' % i)
                else:
                    self.io.cprint('Majorization %d.' % i)
                    i += 1
                assert i <= 50, 'Majorization > 50, failed.'
            else:
                stop_cond = True
                # self.io.cprint('lam_con = 0, majorization skipped.')
    """

    def update_beta(self, X, Y, pair, t0_step=1.5):
        beta_prev = self.model.beta.data.clone()
        grad_con = self.get_Contrastive_grad(X, pair)

        stop_cond = False
        # Initialize t as tr(J J^T) = J.norm()
        t = grad_con.norm() ** 2
        self.io.cprint('Initial t = {:2.12f}.'.format(t))
        i = 1
        while not stop_cond:
            self.update_beta_once(beta_prev, X, Y, pair, t)

            if self.lam_con > 0:
                stop_cond = self.validate_majorization(t, X, pair, grad_con, beta_prev)
                t *= t0_step  # enlarge step-size
                if stop_cond:
                    self.io.cprint('Majorization succeeded (%d).' % i)
                else:
                    self.io.cprint('Majorization %d.' % i)
                    i += 1
                assert i <= 50, 'Majorization > 50, failed.'
            else:
                stop_cond = True
                # self.io.cprint('lam_con = 0, majorization skipped.')

    def update_beta_once(self, beta_prev, X, Y, pair, t=1):
        # Update beta with Majorization Trick, t given
        # return: beta_prev, previous beta for majorization scrutiny
        L_list = self.model.L_list
        gp_idx_list = self.model.gp_idx_list
        lam1, lam2 = self.lam1, self.lam2
        lam_Gau, lam_con = self.lam_Gau, self.lam_con
        if lam_Gau == 0:
            assert L_list[0] is None
        Y = Y.view(-1, 1)
        n = 1

        grad_con = self.get_Contrastive_grad(X, pair)
        self.io.cprint('grad_con norm = {:2.12f}'.format(grad_con.norm()))

        for L, (num_g, idx_g) in zip(L_list, enumerate(gp_idx_list)):
            X_g = X[:, idx_g]
            # self.io.cprint('{} beta norm = {:2.4f}, grad_con norm = {:2.4f}'.format(num_g,
            #                                                                self.model.beta.data[idx_g].norm(),
            #                                                                grad_con.norm()))

            if (self.model.beta.data[idx_g] != 0).sum() > 0:
                # Grad with beta[idx_g] = 0, '_wo_' means 'without'.
                beta_wo_g = nullify(self.model.beta.data, idx_g)
                grad_wo_g = - X_g.t().mm(Y - X.mm(beta_wo_g)) * 1/n
                if lam_con > 0:
                    grad_wo_g += lam_con * (grad_con[idx_g] + t * (0. - beta_prev[idx_g]))

                if group_condition(grad_wo_g, lam1, threshold=lam2 * len(idx_g) ** 0.5) and lam2 > 0:  #
                    # Group Shrinkage
                    self.model.beta.data[idx_g] *= 0.
                    self.io.cprint('Group {} deleted with lam2={}.'.format(num_g, lam2))
                else:
                    # L_norm = L.norm()
                    L_norm = 1.
                    for k in range(len(idx_g)):

                        idx_gk = idx_g[k]
                        X_gk = X[:, idx_gk].view(-1, 1)
                        # Grad with beta[idx_g[k]] = 0
                        beta_wo_gk = nullify(self.model.beta.data, idx_gk)
                        grad_wo_gk = - X_gk.t().mm(Y - X.mm(beta_wo_gk)) * 1/n
                        if lam_con > 0:
                            grad_wo_gk += lam_con * (grad_con[idx_gk] + t * (0. - beta_prev[idx_gk]))
                        if lam_Gau > 0:
                            grad_wo_gk += lam_Gau * L[k, :].view(1, -1).mm(beta_wo_gk[idx_g]) * L_norm

                        # Individual Shrinkage
                        soft_grad_gk = soft((-1) * grad_wo_gk, lam1)
                        normalizer = X_gk.t().mm(X_gk) * 1/n
                        if lam2 > 0:
                            normalizer += lam2 * 1. / self.model.beta.data[idx_g].norm() * len(idx_g)**0.5  #
                        if lam_con > 0:
                            normalizer += lam_con * t
                        if lam_Gau > 0:
                            normalizer += lam_Gau * L[k, k] * L_norm

                        self.model.beta.data[idx_gk] = soft_grad_gk / normalizer

    def get_Contrastive_grad(self, X, pair):
        if self.lam_con == 0:
            return torch.zeros_like(self.model.beta)

        if self.model.beta.grad is not None:
            self.model.beta.grad.zero_()
        loss_con = self.lam_con * self.model.get_Lcon_neg_mask(X, pair)
        loss_con.backward()
        beta_grad_con = self.model.beta.grad

        return beta_grad_con

    def validate_majorization(self, t, X, pair, grad_con, beta_prev):
        beta_new = self.model.beta.data.clone()
        loss_con_orig = self.model.get_Lcon_neg_mask(X, pair) * self.lam_con
        # self.io.cprint('prev beta = {:2.4f}', (beta_prev).norm())
        # self.io.cprint('new beta = {:2.4f}', (beta_new).norm())
        self.model.beta.data = beta_prev
        delta_beta = beta_new - beta_prev
        loss_con_majorized = self.model.get_Lcon_neg_mask(X, pair) * self.lam_con\
                             + grad_con.t().mm(delta_beta) \
                             + t * delta_beta.t().mm(delta_beta)
        self.model.beta.data = beta_new

        return loss_con_orig <= loss_con_majorized


def group_condition(grad, lam1, threshold):
    soft_grad = soft(grad, lam1)
    return soft_grad.norm() <= threshold


def soft(z, lam1):
    delta_z = torch.where(z.abs() < lam1, z, lam1 * z.sign())
    delta_z.to(z.device)
    return z - delta_z


def nullify(beta, idx_g):
    beta = beta.clone()
    beta[idx_g] *= 0.
    return beta


def Gaussian_matrix(loc, p=2):
    assert p >= 1, 'order p should be larger than 1.'

    d = len(loc)
    G = np.ones([d, d])
    sigma = p_norm(loc - np.mean(loc), ord=p) / (d ** (1 / p))  # Generalized p-order Standard Deviation

    for i in range(d):
        for j in range(i):
            G[i, j] = np.exp(- np.abs(loc[i] - loc[j]) ** p / sigma ** p)
            G[j, i] = G[i, j]

    G = G / d

    return G

import copy
import random
import numpy as np
import scipy.sparse as sp


'''
    Augmentations
'''

def dropout_adj(coo_adj, p):
    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))
    coo_adj = coo_adj.tocoo()
    col = list(coo_adj.col.reshape(-1))
    row = list(coo_adj.row.reshape(-1))
    obs = np.random.binomial(1, 1-p, len(col))
    mask = sp.coo_matrix((obs, (row, col)), shape=(coo_adj.shape[0], coo_adj.shape[0]))
    drop_edge_adj = coo_adj.multiply(mask)
    return drop_edge_adj

def aug_feature_dropout(input_feat, drop_percent = 0.2):
    aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S


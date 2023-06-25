import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import time

import utils.aug as aug
from utils import process
from modules.clustering import cluster_memory



# def train(args, model, adj, diff, features, local_memory_embeddings, sparse, device, optimizer):
#
#     model.train()
#     cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
#
#     t0 = time.time()
#     aug_adj1 = aug.dropout_adj(adj, p=args.drop_edge)
#     aug_adj2 = sp.csr_matrix(diff)
#     features = features.squeeze(0)
#     aug_features1 = aug.aug_feature_dropout(features, args.drop_feat1)
#     aug_features2 = aug.aug_feature_dropout(features, args.drop_feat2)
#     aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
#     aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))
#     t1 = time.time()
#     print('aug cost', t1 - t0)
#
#     t0 = time.time()
#     if sparse:
#         adj_1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1).to(device)
#         adj_2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2).to(device)
#     else:
#         aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
#         aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()
#         adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(device)
#         adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(device)
#
#     aug_features1 = aug_features1.to(device)
#     aug_features2 = aug_features2.to(device)
#     t1 = time.time()
#     print('sparse cost', t1 - t0)
#
#     start = time.time()
#
#     t0 = time.time()
#     # 建立assignments，即构建分类任务
#     # print(local_memory_embeddings.size())
#     assignments = cluster_memory(args, model, local_memory_embeddings, features.shape[-2])
#     t1 = time.time()
#     print('cluster cost', t1 - t0)
#
#     t0 = time.time()
#     # 得到embed和prototypes
#     embed_1, prototypes_1 = model(aug_features1, adj_1, sparse)
#     embed_2, prototypes_2 = model(aug_features2, adj_2, sparse)
#     embed_2 = embed_2.detach()
#     embed_1 = embed_1.detach()
#     t1 = time.time()
#     print('model cost', t1 - t0)
#
#     t0 = time.time()
#     # 计算loss
#     loss = 0
#     for h in range(len(args.nmb_prototypes)):
#         scores_1 = prototypes_1[h] / args.temperature
#         scores_2 = prototypes_2[h] / args.temperature
#         scores = torch.cat((scores_1, scores_2))
#         targets = assignments[h][:].repeat(sum(args.nmb_crops)).cuda(non_blocking=True)
#         loss += cross_entropy(scores, targets)
#     loss /= len(args.nmb_prototypes)
#     t1 = time.time()
#     print('loss cost', t1 - t0)
#
#
#     t0 = time.time()
#     # ============ backward and optim step ... ============
#     optimizer.zero_grad()
#     loss.backward()
#     # cancel some gradients
#     t1 = time.time()
#     print('opt cost', t1 - t0)
#
#     end = time.time()
#     print('total cost ', end - start)
#     print('===========================================')
#
#     for name, p in model.named_parameters():
#         if "prototypes" in name:
#             p.grad = None
#     optimizer.step()
#
#
#     embed_1 = embed_1.unsqueeze(dim=0)
#     embed_2 = embed_2.unsqueeze(dim=0)
#
#     # 更新memory_bank
#     local_memory_embeddings = torch.cat((embed_1, embed_2), dim=0)
#
#     return loss, local_memory_embeddings

def train(args, model, adj, diff, features, local_memory_embeddings, sparse, device, optimizer):

    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    # 建立assignments，即构建分类任务
    # print(local_memory_embeddings.size())
    assignments = cluster_memory(args, model, local_memory_embeddings, features.shape[-2])

    aug_adj1 = aug.dropout_adj(adj, p=args.drop_edge)
    aug_adj2 = sp.csr_matrix(diff)
    features = features.squeeze(0)
    aug_features1 = aug.aug_feature_dropout(features, args.drop_feat1)
    aug_features2 = aug.aug_feature_dropout(features, args.drop_feat2)
    t0 = time.time()
    aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
    aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))
    t1 = time.time()

    if sparse:
        adj_1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1).to(device)
        adj_2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2).to(device)
    else:
        aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
        aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()
        adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(device)
        adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(device)

    aug_features1 = aug_features1.to(device)
    aug_features2 = aug_features2.to(device)

    # 得到embed和prototypes
    embed_1, prototypes_1 = model(aug_features1, adj_1, sparse)
    embed_2, prototypes_2 = model(aug_features2, adj_2, sparse)
    embed_2 = embed_2.detach()
    embed_1 = embed_1.detach()

    # 计算loss
    loss = 0
    for h in range(len(args.nmb_prototypes)):
        scores_1 = prototypes_1[h] / args.temperature
        scores_2 = prototypes_2[h] / args.temperature
        scores = torch.cat((scores_1, scores_2))
        targets = assignments[h][:].repeat(sum(args.nmb_crops)).cuda(non_blocking=True)
        loss += cross_entropy(scores, targets)
    loss /= len(args.nmb_prototypes)

    # ============ backward and optim step ... ============
    optimizer.zero_grad()
    loss.backward()

    # cancel some gradients
    for name, p in model.named_parameters():
        if "prototypes" in name:
            p.grad = None
    optimizer.step()


    embed_1 = embed_1.unsqueeze(dim=0)
    embed_2 = embed_2.unsqueeze(dim=0)

    # 更新memory_bank
    local_memory_embeddings = torch.cat((embed_1, embed_2), dim=0)

    return loss, local_memory_embeddings

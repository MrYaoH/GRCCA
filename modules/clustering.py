import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

import utils.aug as aug
from utils import process

def init_memory(args, model, adj, diff, features, sparse, device):

    aug_adj1 = aug.dropout_adj(adj, p=args.drop_edge)
    aug_adj2 = sp.csr_matrix(diff)
    features = features.squeeze(0)
    aug_features1 = aug.aug_feature_dropout(features, args.drop_feat1)
    aug_features2 = aug.aug_feature_dropout(features, args.drop_feat2)
    aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
    aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

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

    embed_1 = model(aug_features1, adj_1, sparse)
    embed_2 = model(aug_features2, adj_2, sparse)

    local_memory_embeddings = torch.cat((embed_1, embed_2), dim=0)

    return local_memory_embeddings

def cluster_memory(args, model, local_memory_embeddings, size_dataset, nmb_kmeans_iters=10):
    j = 0
    assignments = -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long() 

    with torch.no_grad():
        for i_K, K in enumerate(args.nmb_prototypes):
            # run K_means

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, args.proj_out).cuda()

            random_idx = torch.randperm(len(local_memory_embeddings[j]))[:K]
            assert len(random_idx) >= K, "please reduce the number of centroids"
            centroids = local_memory_embeddings[j][random_idx][:]

            for n_iter in range(nmb_kmeans_iters + 1):

                # E step
                dot_products = torch.mm(local_memory_embeddings[j], centroids.t())    
                _, local_assignments = dot_products.max(dim=1)    

                # finish
                if n_iter == nmb_kmeans_iters:
                    break

                # M step
                where_helper = get_indices_sparse(local_assignments.cpu().numpy())

                counts = torch.zeros(K).cuda().int()
                emb_sums = torch.zeros(K, args.proj_out).cuda()
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(
                            local_memory_embeddings[j][where_helper[k][0]],
                            dim=0,
                        )
                    counts[k] = len(where_helper[k][0])
                mask = counts > 0
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1) 

                # normalize centroids
                centroids = nn.functional.normalize(centroids, dim=1, p=2)

            getattr(model.prototypes, "prototypes" + str(i_K)).weight.copy_(centroids)
            assignments[i_K] = local_assignments
            j = (j + 1) % len(args.crops_for_assign)

    return assignments

def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = sp.csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]

import torch
import numpy as np
import scipy.sparse as sp
import random
import os

'''
    Dataset constructor for non-citation networks
'''

def get_train_val_test_split(split_start, split_len, dataset='amazon_electronics_photo', path=None):
    path = 'data/{}.npz'.format(dataset)
    data_npz = np.load(path)
    attribute = sp.csr_matrix((data_npz['attr_data'], data_npz['attr_indices'], data_npz['attr_indptr']),
                      data_npz['attr_shape']).todense()     # attribute

    attribute = torch.from_numpy(attribute).to(torch.float)
    attribute[attribute > 0] = 1
    attribute = sp.lil_matrix(attribute)

    # Get adjacent matrix
    adj = sp.csr_matrix((data_npz['adj_data'], data_npz['adj_indices'], data_npz['adj_indptr']),
                        data_npz['adj_shape']).tocoo()  # adj

    is_undirect = (adj != adj.T).nnz == 0
    if ~is_undirect:
        if is_weighted(adj.tocsr()):
            raise ValueError("Convert to unweighted graph first.")
        else:
            print('transform the directed graph to be undirected')
            adj = adj + adj.T
            adj[adj != 0] = 1

    k = adj.toarray().diagonal()
    is_non = (k.sum() == 0)
    if ~is_non:
        adj = eliminate_self_loops(adj)

    # Get labels in one-hot format
    y = torch.from_numpy(data_npz['labels']).to(torch.long)
    split_dependencies = np.array(data_npz['labels'])
    labels = to_one_hot(split_dependencies, dimension=y.unique().size(0))

    train_idx = []
    val_idx = []
    test_idx = []

    if os.path.exists('data/split_{}_train_idx.npy'.format(dataset)):
        print('the split existed, so we directly load this splited data')
        train_idx = np.load('data/split_{}_train_idx.npy'.format(dataset), allow_pickle=True)
        val_idx = np.load('data/split_{}_val_idx.npy'.format(dataset), allow_pickle=True)
        test_idx = np.load('data/split_{}_test_idx.npy'.format(dataset), allow_pickle=True)
    else:
        for i in y.unique():
            each_class_candi = [k for k, x in enumerate(split_dependencies) if x == i.item()]
            random.shuffle(each_class_candi)

            # Build training set
            train_idx.extend(each_class_candi[split_start: split_start + split_len])

            # Build validation set
            new_start = split_start + split_len
            val_idx.extend(each_class_candi[new_start:new_start + split_len])
            test_idx.extend(each_class_candi[new_start + split_len:len(each_class_candi)])

        print('we contruct this splited data')
        np.save('data/split_{}_train_idx'.format(dataset), train_idx)
        np.save('data/split_{}_val_idx'.format(dataset), val_idx)
        np.save('data/split_{}_test_idx'.format(dataset), test_idx)

    return adj, attribute, labels, train_idx, val_idx, test_idx


def remove_self_loops(edge_index):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index

def is_weighted(adj_matrix):
    """Check if the graph is weighted (edge weights other than 1)."""
    '''
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
    '''
    return np.any(np.unique(adj_matrix[adj_matrix != 0].A1) != 1)

def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def to_one_hot(label, dimension=46):
    '''
        One-hot encoding transformer
    '''
    results = np.zeros((len(label), dimension))
    for i, label in enumerate(label):
        results[i, label] = 1.
    return results
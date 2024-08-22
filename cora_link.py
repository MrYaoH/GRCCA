import numpy as np
import scipy.sparse as sp
import torch
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from utils import process
from utils import aug
from scipy.sparse import csr_matrix
from utils.utils import str_to_bool
from modules.encoder import GraphEncoder
from modules.model import GRCCA
from test import link_eva
from modules.clustering import init_memory
from train import train

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1908)
parser.add_argument('--data', type=str, default='cora')
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--sparse', type=str_to_bool, default=True)
parser.add_argument('--gnn_dim', type=int, default=256)
parser.add_argument('--proj_dim', type=int, default=256)
parser.add_argument('--proj_hid', type=int, default=256)
parser.add_argument('--proj_out', type=int, default=256)
parser.add_argument('--proto_dim', type=int, default=256)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument("--nmb_prototypes", default=[14, 14, 14, 14], type=int, nargs="+",
                    help="number of prototypes - it can be multihead")
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")

parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--drop_edge', type=float, default=0.2)
parser.add_argument('--drop_feat1', type=float, default=0.3)
parser.add_argument('--drop_feat2', type=float, default=0.4)
parser.add_argument("--temperature", default=0.05, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")

parser.add_argument('--test_rate', dest='test_rate', type=float, default=0.1, help='')


args = parser.parse_args()
torch.set_num_threads(4)

def main():

    # Set the seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Set hyper-parameters
    dataset = args.data
    gnn_output_size = args.gnn_dim
    projection_size = args.proj_dim
    projection_hidden_size = args.proj_hid
    projection_out_size = args.proj_out
    nmb_prototpyes = args.nmb_prototypes
    alpha = args.alpha
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    patience = args.patience
    sparse = args.sparse
    act = args.act

    # Load datasets
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    adj_sparse = adj
    if os.path.exists('process_dataset/split_{}_adj_train.npz'.format(dataset)):
        print('the split existed, so we directly load this splited data')
        mtr = np.load('process_dataset/split_{}_adj_train.npz'.format(dataset), allow_pickle=True)
        adj_train = csr_matrix((mtr['data'], (mtr['row'], mtr['col'])))
        test_edges = np.load('process_dataset/split_{}_test_edges.npy'.format(dataset), allow_pickle=True)
        test_edges_false = np.load('process_dataset/split_{}_test_edges_false.npy'.format(dataset), allow_pickle=True)
    else:
        print('we contruct this splited data')
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = process.mask_test_edges(adj, test_frac=args.test_rate, val_frac=0.05)
        mtr = adj_train.tocoo()
        np.savez('process_dataset/split_{}_adj_train'.format(dataset), data=mtr.data, row=mtr.row, col=mtr.col)
        np.save('process_dataset/split_{}_test_edges'.format(dataset), test_edges)
        np.save('process_dataset/split_{}_test_edges_false'.format(dataset), test_edges_false)

    adj = adj_train

    if os.path.exists('data/diff_{}_{}_link.npy'.format(dataset, alpha)):
        diff = np.load('data/diff_{}_{}_link.npy'.format(dataset, alpha), allow_pickle=True)
    else:
        diff = aug.gdc(adj, alpha=alpha, eps=0.0001)
        np.save('data/diff_{}_{}_link.npy'.format(dataset, alpha), diff)

    # features, _ = process.preprocess_features(features)
    features = features.todense()
    features = torch.FloatTensor(features[np.newaxis])  # 将tensor扩展一个维度
    input_size = features.shape[-1]

    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    norm_diff = sp.csr_matrix(diff)
    if sparse:
        eval_adj = process.sparse_mx_to_torch_sparse_tensor(norm_adj)
        eval_diff = process.sparse_mx_to_torch_sparse_tensor(norm_diff)
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = torch.FloatTensor(eval_adj[np.newaxis])
        eval_diff = torch.FloatTensor(eval_diff[np.newaxis])

    # Record results over each run
    AUC = []
    AP = []

    for i in range(args.runs):
        # Bulid model
        model = GraphEncoder(input_size, gnn_output_size, act='relu')
        cgcn = GRCCA(encoder=model,
                     projection_input_size=projection_size,
                     projection_hidden_size=projection_hidden_size,
                     projection_output_size=projection_out_size,
                     nmb_prototypes=nmb_prototpyes).to(device)

        # Set optimizer
        opt = torch.optim.Adam(cgcn.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize memory
        local_memory_embeddings = init_memory(args, model, adj, diff, features, sparse, device)

        # Training
        best = 1e9
        patience_count = 0

        for epoch in range(epochs):

            loss, local_memory_embeddings = train(args, cgcn, adj, diff, features, local_memory_embeddings, sparse,
                                                  device, opt)
            print(f'(T) | Run={i:03d}, Epoch={epoch:03d}, loss={loss:.4f}')
            if loss < best:
                best = loss
                patience_count = 0
                # print("hell")
                torch.save(model.state_dict(), dataset + '-link.pkl')

            else:
                patience_count += 1

            if patience_count == patience:
                # print('Early stopping!')
                break
        # 输出最优结果
        sc_roc, sc_ap = link_eva(eval_adj, eval_diff, features, dataset, sparse, input_size, gnn_output_size, test_edges, test_edges_false, adj_sparse, act)
        AUC.append(sc_roc * 100)
        AP.append(sc_ap * 100)

    print('-----------------------------------------')
    print('Average AUC:')

    print(np.mean(AUC))
    print(np.std(AUC))

    print('-----------------------------------------')
    print('Average AP:')

    print(np.mean(AP))
    print(np.std(AP))

if __name__ == '__main__':
    main()









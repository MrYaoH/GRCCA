import os
import time
import torch
import argparse
import numpy as np
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

import utils.aug as aug
from utils import process
from utils.utils import str_to_bool, setting_seed, plot_loss
from modules.encoder import GraphEncoder
from modules.model import GRCCA
from modules.clustering import init_memory
from train import train
from test import evaluate
from utils.other_process import get_train_val_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1908)
parser.add_argument('--data', type=str, default='ms_academic_cs')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--sparse', type=str_to_bool, default=True)
parser.add_argument('--gnn_dim', type=int, default=256)
parser.add_argument('--proj_dim', type=int, default=256)
parser.add_argument('--proj_hid', type=int, default=256)
parser.add_argument('--proj_out', type=int, default=256)
parser.add_argument('--proto_dim', type=int, default=256)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument("--nmb_prototypes", default=[24, 24, 24, 24], type=int, nargs="+",
                    help="number of prototypes - it can be multihead")
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")

parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--drop_edge', type=float, default=0.4)
parser.add_argument('--drop_feat1', type=float, default=0.2)
parser.add_argument('--drop_feat2', type=float, default=0.2)
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")

args = parser.parse_args()
torch.set_num_threads(4)

def main():

    # Set the seed
    setting_seed(args.seed)
    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Set hyper-parameters
    eval_every_epoch = args.eval_every
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
    adj, features, labels, idx_train, idx_val, idx_test = get_train_val_test_split(0, 30, dataset)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    input_size = features.shape[-1]

    # Pre-process the graph diffusion process
    if os.path.exists('data/diff_{}_{}.npy'.format(dataset, alpha)):
        diff = np.load('data/diff_{}_{}.npy'.format(dataset, alpha), allow_pickle=True)
    else:
        diff = aug.gdc(adj, alpha=alpha, eps=0.0001)
        np.save('data/diff_{}_{}'.format(dataset, alpha), diff)

    features = features.todense()

    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

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

    # Build model
    model = GraphEncoder(input_size, gnn_output_size, act=act)
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
    best = 0
    patience_count = 0

    results = []
    loss_pool = []
    for epoch in range(epochs):
        # Count the time cost
        start = time.time()
        loss, local_memory_embeddings = train(args, cgcn, adj, diff, features, local_memory_embeddings, sparse, device, opt)
        epoch_train_time = time.time() - start
        loss_pool.append(loss.item())

        # print('Time cost is: ', epoch_train_time)
        if epoch % eval_every_epoch == 0:
            acc = evaluate(eval_adj, eval_diff, features, model, idx_train, idx_test, sparse, input_size, gnn_output_size, labels, act)
            if acc > best:
                best = acc
                patience_count = 0
            else:
                patience_count += 1
            results.append(acc)
            print('\t epoch {:03d} | loss {:.5f} | clf test acc {:.5f}'.format(epoch, loss.item(), acc))
            if patience_count >= patience:
                print('Early Stopping.')
                break

    # Survey loss curve
    # plot_loss(loss_pool, dataset)

    print('\t best acc {:.5f}'.format(max(results)))
    return max(results)

if __name__ == '__main__':
    result = main()
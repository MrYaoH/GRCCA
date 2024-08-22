import torch
import numpy as np
from modules.encoder import GraphEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

def evaluate(adj, diff, feat, gnn, idx_train, idx_test, sparse, input_size, gnn_output_size, labels, act):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    model = GraphEncoder(input_size, gnn_output_size, act=act)  # 1-layer
    model.load_state_dict(gnn.state_dict())
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse)
        embeds2 = model(feat, diff, sparse)
        train_embs = embeds1[0, idx_train] + embeds2[0, idx_train]
        test_embs = embeds1[0, idx_test] + embeds2[0, idx_test]
        train_labels = torch.argmax(labels[0, idx_train], dim=1)
        test_labels = torch.argmax(labels[0, idx_test], dim=1)
    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    return accuracy_score(test_labels, pred_test_labels)

def link_eva(adj, diff, feat, dataset, sparse, input_size, gnn_output_size, test_edges, test_edges_false, adj_sparse, act):
    model = GraphEncoder(input_size, gnn_output_size, act=act)  # 1-layer
    model.load_state_dict(torch.load(dataset+'-link.pkl'))
    embeds1 = model(feat, adj, sparse)
    embeds2 = model(feat, diff, sparse)
    embeds = embeds1 + embeds2
    embs = embeds[0, :]
    embs = embs / embs.norm(dim=1)[:, None]
    sc_roc, sc_ap = get_roc_score(test_edges, test_edges_false, embs.cpu().detach().numpy(), adj_sparse)
    print('AUC', sc_roc, 'AP', sc_ap)
    return sc_roc, sc_ap

def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"

    score_matrix = np.dot(embeddings, embeddings.T)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        pos.append(adj_sparse[edge[0], edge[1]])  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        neg.append(adj_sparse[edge[0], edge[1]])  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score
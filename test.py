import torch

from modules.encoder import GraphEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
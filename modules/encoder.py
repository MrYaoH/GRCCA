import torch
import torch.nn as nn

class GCNLayer(nn.Module):

    def __init__(self, in_ft, out_ft, act='prelu', bias=True):

        super(GCNLayer, self).__init__()

        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

class GraphEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 act='prelu',
                 gcn=GCNLayer):

        super().__init__()

        self.gnn = gcn(input_size, output_size, act)

    def forward(self, in_feat, adj, sparse=True):

        in_feat = self.gnn(in_feat, adj, sparse)
        return in_feat

class MLP(nn.Module):
    def __init__(self, inp_size, out_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.net(x)
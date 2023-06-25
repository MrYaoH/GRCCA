import torch.nn as nn
from modules.encoder import MLP

class GRCCA(nn.Module):
    def __init__(self,
                 encoder,
                 projection_input_size,
                 projection_hidden_size,
                 projection_output_size,
                 nmb_prototypes):
        super().__init__()
        self.encoder = encoder
        self.projector = MLP(projection_input_size, projection_output_size, projection_hidden_size)
        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(projection_output_size, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(projection_output_size, nmb_prototypes, bias=False)


    def forward(self, in_feat, adj, sparse):
        representations = self.encoder(in_feat, adj, sparse)
        representations = representations.view(-1, representations.size(-1))
        proj = self.projector(representations)
        p_prototypes = self.prototypes(proj)
        return proj, p_prototypes

    def encode(self, in_feat, adj, sparse):
        representations = self.encoder(in_feat, adj, sparse)
        representations = representations.view(-1, representations.size(-1))
        proj = self.projector(representations)
        return proj

    def prototype(self, proj_emb):
        p_prototypes = self.prototypes(proj_emb)
        return p_prototypes


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))    # Save prototype

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))   # Calculate similarity
        return out
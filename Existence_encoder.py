import torch
import torch.nn as nn
from Graph_encoder import graph_encoder
from contrast import Contrast

class existence_encoder(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, tau, lam, pmi_rel_names,emb_rel_names):
        super(existence_encoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pmi_encoder = graph_encoder(n_inp,  n_hid, n_out, pmi_rel_names).to(self.device)
        self.emb_encoder = graph_encoder(n_inp,  n_hid, n_out,  emb_rel_names).to(self.device)
        self.contrast = Contrast(n_out, tau, lam).to(self.device)

    def forward(self, pos, g_pmi, g_emb, node_features):
        z_pmi = self.pmi_encoder(g_pmi, node_features)['file'].to(self.device)
        z_emb = self.emb_encoder(g_emb, node_features)['file'].to(self.device)
        loss = self.contrast(z_pmi, z_emb, pos)
        return z_pmi,z_emb, loss



import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from core.config import cfg as config

class ResLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(x) + x

class CompoundCFG(nn.Module):
    def __init__(self, V, NT, T,
                 h_dim=512,
                 w_dim=512,
                 z_dim=64,
                 s_dim=256):
        super(CompoundCFG, self).__init__()
        assert z_dim >= 0
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.root_emb = nn.Parameter(torch.randn(1, s_dim))
        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))

        self.rule_mlp = nn.Linear(s_dim + z_dim, self.NT_T ** 2)
        self.root_mlp = nn.Sequential(nn.Linear(s_dim + z_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))
        self.term_mlp = nn.Sequential(nn.Linear(s_dim + z_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        if z_dim > 0:
            self.enc_emb = nn.Embedding(V, w_dim)
            self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.enc_out = nn.Linear(h_dim * 2, z_dim * 2)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict)

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, l):
        x_embbed = self.enc_emb(x)
        # self.enc_rnn.flatten_parameters()
        packed_x_embbed = pack_padded_sequence(x_embbed, l.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.enc_rnn(packed_x_embbed)
        unpacked_h = pad_packed_sequence(h, batch_first=True, padding_value=float('-inf'))[0]
        out = self.enc_out(unpacked_h.max(1)[0])

        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim:]
        return mean, lvar

    def forward(self, x, l):
        b, n = x.shape[:2]
        if self.z_dim > 0:
            mean, lvar = self.enc(x, l)
            kl = self.kl(mean, lvar).sum(1)
            z = mean
            if self.training: # NOTE: use mean value during evaluation
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob

        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            )
            if self.z_dim > 0:
                z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                    b, n, self.T, self.z_dim
                )
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim
            )
            if self.z_dim > 0:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl

class GatedEmbedding(nn.Module):
    def __init__(self, cfg):
        super(GatedEmbedding, self).__init__()
        self.cfg = cfg.MODEL.PARAMS
        self.gated_embeddings = nn.ModuleList()
        for expert in config.DATASET.EXPERTS:
            self.gated_embeddings.append(nn.Linear(self.cfg.sem_dim, self.cfg.sem_dim))

    def forward(self, captions):
        outs = []
        for linear in self.gated_embeddings:
            z = linear(captions)
            z = z * torch.sigmoid(z)
            z = F.normalize(z, dim=-1)
            outs.append(z)
        outs = torch.stack(outs, dim=1)
        return outs


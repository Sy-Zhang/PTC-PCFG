import torch
import torch.nn.functional as F
from dataset.datasets.pentathlon_dataset import UNK

class LSTMEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super(LSTMEncoder, self).__init__()
        self.cfg = cfg.SPAN_ENCODER.PARAMS
        self.NT = cfg.MODEL.PARAMS.nt_states
        self.sem_dim = cfg.MODEL.PARAMS.sem_dim
        self.enc_rnn = torch.nn.LSTM(self.cfg.word_dim, self.cfg.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.enc_out = torch.nn.Linear(self.cfg.lstm_dim * 2, self.NT * self.sem_dim)
        self.enc_emb = torch.nn.Embedding(cfg.MODEL.PARAMS.vocabulary_size, self.cfg.word_dim, padding_idx=UNK)

    def forward(self, tokens, captions, caption_lengths):
        """
        lstm over every span, a.k.a. segmental rnn
        """
        lengths = caption_lengths.cuda()
        x_emb = self.enc_emb(captions)
        b, N, dim = x_emb.size()
        word_mask = torch.arange(0, N, device=x_emb.device).unsqueeze(0).expand(b, N).long()
        max_len = lengths.unsqueeze(-1).expand_as(word_mask)
        word_mask = word_mask < max_len
        word_vect = x_emb * word_mask.unsqueeze(-1)
        feats = torch.zeros(b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=x_emb.device)
        beg_idx = 0
        for k in range(1, N):
            inc = torch.arange(N - k, device=x_emb.device).view(N - k, 1)  # .expand(N - k, k + 1)
            idx = torch.arange(k + 1, device=x_emb.device).view(1, k + 1).repeat(N - k, 1)
            idx = (idx + inc).view(-1)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(b, -1, dim)

            feat = torch.gather(word_vect, 1, idx)
            feat = feat.view(b, N - k, k + 1, dim)
            feat = feat.view(-1, k + 1, dim)
            self.enc_rnn.flatten_parameters()
            feat = self.enc_out(self.enc_rnn(feat)[0])
            feat = feat.view(b, N - k, k + 1, self.NT, self.sem_dim)
            feat = F.normalize(feat.sum(2), dim=-1)
            end_idx = beg_idx + N - k
            feats[:, beg_idx: end_idx] = feat
            beg_idx = end_idx
        return feats
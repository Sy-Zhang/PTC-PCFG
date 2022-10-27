import torch
from torch import nn
import torch.nn.functional as F
from .s3dg.s3dg import Sentence_Embedding
from collections import OrderedDict
from nltk.tokenize.treebank import TreebankWordDetokenizer as detokenizer

class S3DGEncoder(Sentence_Embedding):
    def __init__(self, cfg):
        super(S3DGEncoder, self).__init__(512, token_to_word_path='.cache/howto100m/s3d_dict.npy')
        self.detokenizer = detokenizer()
        self.cfg = cfg.SPAN_ENCODER.PARAMS
        self.max_words = cfg.DATASET.MAX_TEXT_LENGTH
        self.NT = cfg.MODEL.PARAMS.nt_states
        self.sem_dim = cfg.MODEL.PARAMS.sem_dim
        state_dict = OrderedDict({k[12:]: v for k, v in torch.load(self.cfg.checkpoint).items() if 'text_module' in k})
        self.load_state_dict(state_dict)
        self.fc2_nt = nn.Linear(self.fc2.in_features, self.NT*self.fc2.out_features)
        self.fc2_nt.weight = nn.Parameter(torch.cat([self.fc2.weight for _ in range(self.NT)], dim=0), requires_grad=True)
        self.fc2_nt.bias = nn.Parameter(torch.cat([self.fc2.bias for _ in range(self.NT)], dim=0), requires_grad=True)
        for param in self.parameters():
            param.requires_grad = self.cfg.finetune

    def forward(self, tokens, captions, caption_lengths):
        lengths = caption_lengths.cuda()
        sentences = [self.detokenizer.detokenize(t) for t in tokens]
        self.max_words = max(len(t) for t in tokens)
        x = self._words_to_ids(sentences).to(caption_lengths.device)
        x_emb = self.word_embd(x)
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

            x = F.relu(self.fc1(feat))
            x = torch.max(x, dim=1)[0]
            feat = self.fc2_nt(x)

            feat = feat.view(b, N - k, self.NT, self.sem_dim)
            end_idx = beg_idx + N - k
            feats[:, beg_idx: end_idx] = feat
            beg_idx = end_idx
        return feats

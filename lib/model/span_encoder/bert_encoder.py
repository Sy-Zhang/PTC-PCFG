import torch
from torch import nn
import torch.nn.functional as F
from .tinybert.tokenization import BertTokenizer
from .tinybert.modeling import TinyBertForPreTraining
from nltk.tokenize.treebank import TreebankWordDetokenizer as detokenizer

class BERTEncoder(nn.Module):
    def __init__(self, cfg):
        super(BERTEncoder, self).__init__()
        self.cfg = cfg.SPAN_ENCODER.PARAMS
        self.tokenizer = BertTokenizer.from_pretrained('.cache/tinybert', do_lower_case=True)
        self.bert = TinyBertForPreTraining.from_pretrained('.cache/tinybert').bert
        self.detokenizer = detokenizer()
        for param in self.bert.parameters():
            param.requires_grad = False

        self.NT = cfg.MODEL.PARAMS.nt_states
        self.sem_dim = cfg.MODEL.PARAMS.sem_dim
        self.text_projector = nn.Linear(312, self.NT*self.sem_dim)

    def forward(self, tokens, captions, caption_lengths):
        """
        lstm over every span, a.k.a. segmental transformer
        """
        b, N = len(tokens), max([len(t) for t in tokens])
        feats = torch.zeros(b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=caption_lengths.device)
        beg_idx = 0
        for k in range(1, N):
            input_ids = []
            for tok in tokens:
                for i in range(N-k):
                    temp_tokens = self.tokenizer.tokenize(self.detokenizer.detokenize(tok[i:i+k+1]))
                    ids = self.tokenizer.convert_tokens_to_ids(temp_tokens)
                    # if len(ids) < k:
                    #     ids.extend([self.tokenizer.vocab['[PAD]'] for _ in range(k - len(ids))])
                    input_ids.append(torch.LongTensor(ids))
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.vocab['[PAD]'])
            input_mask = (input_ids > 0).long()
            input_mask[:,0] = 1
            x = self.bert(input_ids.to(caption_lengths.device), attention_mask=input_mask.to(caption_lengths.device))[-1]
            x = self.text_projector(x)
            feat = x.view(b, N - k, self.NT, self.sem_dim)
            feat = F.normalize(feat, dim=-1)
            end_idx = beg_idx + N - k
            feats[:, beg_idx: end_idx] = feat
            beg_idx = end_idx
        return feats
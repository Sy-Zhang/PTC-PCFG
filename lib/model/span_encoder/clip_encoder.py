import torch
from torch import nn
from .clip.model import Transformer, LayerNorm
from .clip.clip import tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch.nn.functional as F

class CLIPEncoder(nn.Module):

    def __init__(self, cfg):
        super(CLIPEncoder, self).__init__()
        self.cfg = cfg.SPAN_ENCODER.PARAMS
        self.detokenizer = TreebankWordDetokenizer()

        state_dict = torch.jit.load(self.cfg.checkpoint).state_dict()

        embed_dim = state_dict["text_projection"].shape[1]
        self.context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.load_state_dict(state_dict, strict=False)

        self.NT = cfg.MODEL.PARAMS.nt_states
        self.sem_dim = cfg.MODEL.PARAMS.sem_dim
        assert self.sem_dim == embed_dim
        self.text_projection = nn.Parameter(torch.cat([self.text_projection for _ in range(self.NT)], dim=1))
        for params in self.parameters():
            params.requires_grad = False
        self.text_projection.requires_grad = True

    def encode_text(self, text):
        x = self.token_embedding(text).to(text.device)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(text.device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(text.device)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, tokens, captions, caption_lengths):
        """
        lstm over every span, a.k.a. segmental transformer
        """
        b, N = len(tokens), max([len(t) for t in tokens])
        feats = torch.zeros(b, int(N * (N - 1) / 2), self.NT, self.sem_dim, device=caption_lengths.device)
        beg_idx = 0
        for k in range(1, N):

            # inc = torch.arange(N - k, device=caption_lengths.device).view(N - k, 1)  # .expand(N - k, k + 1)
            # idx = torch.arange(k + 1, device=caption_lengths.device).view(1, k + 1).repeat(N - k, 1)
            # idx = (idx + inc).view(-1)
            phrases = []
            for tok in tokens:
                for i in range(N-k):
                    phrases.append(self.detokenizer.detokenize(tok[i:i+k+1]))
            text_inputs = torch.cat([tokenize(p) for p in phrases]).to(caption_lengths.device)
            x = self.encode_text(text_inputs)
            feat = x.view(b, N - k, self.NT, self.sem_dim)
            end_idx = beg_idx + N - k
            feats[:, beg_idx: end_idx] = feat
            beg_idx = end_idx
        if self.cfg.normalize:
            feats = F.normalize(feats, dim=-1)
        return feats
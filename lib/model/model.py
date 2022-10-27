import torch
from torch import nn
from torch_struct import SentCFG
import torch.nn.functional as F
from core.config import cfg as config
from .module import CompoundCFG
from .utils import *
from .video_encoder.encoder_factory import get_encoder as get_video_encoder
from .span_encoder.encoder_factory import get_encoder as get_span_encoder
import model.loss as loss

class Random(nn.Module):
    def __init__(self, cfg):
        super(Random, self).__init__()

    def forward(self, inference, tokens, captions, caption_lengths, *videos):
        spans = [get_spans(get_random_tree(l)) for l in caption_lengths.tolist()]
        if inference:
            return spans
        else:
            fake_loss = torch.tensor(0.0, dtype=torch.float, device=captions.device)
            return spans, fake_loss, fake_loss, fake_loss, fake_loss

class LeftBranching(nn.Module):
    def __init__(self, cfg):
        super(LeftBranching, self).__init__()

    def forward(self, inference, tokens, captions, caption_lengths, *videos):
        spans = [get_spans(get_left_branching_tree(l)) for l in caption_lengths.tolist()]
        if inference:
            return spans
        else:
            fake_loss = torch.tensor(0.0, dtype=torch.float, device=captions.device)
            return spans, fake_loss, fake_loss, fake_loss, fake_loss

class RightBranching(nn.Module):
    def __init__(self, cfg):
        super(RightBranching, self).__init__()

    def forward(self, inference, tokens, captions, caption_lengths, *videos):
        spans = [get_spans(get_right_branching_tree(l)) for l in caption_lengths.tolist()]
        if inference:
            return spans
        else:
            fake_loss = torch.tensor(0.0, dtype=torch.float, device=captions.device)
            return spans, fake_loss, fake_loss, fake_loss, fake_loss

class CPCFGs(nn.Module):
    def __init__(self, cfg):
        super(CPCFGs, self).__init__()
        self.cfg = cfg.MODEL.PARAMS
        self.vse_lm_alpha = self.cfg.vse_lm_alpha
        self.parser = CompoundCFG(
            self.cfg.vocabulary_size, self.cfg.nt_states, self.cfg.t_states,
            h_dim = self.cfg.h_dim,
            w_dim = self.cfg.w_dim,
            z_dim = self.cfg.z_dim,
            s_dim = self.cfg.s_dim
        )

    def forward_parser(self, captions, lengths):
        params, kl = self.parser(captions, lengths)
        dist = SentCFG(params, lengths=lengths)

        the_spans = dist.argmax[-1]
        argmax_spans, trees, lprobs = extract_parses(the_spans, lengths.tolist(), inc=0)

        ll = dist.partition
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        return nll, kl, argmax_spans, trees, lprobs


    def forward(self, inference, tokens, captions, caption_lengths, *videos):
        nll, kl, argmax_spans, trees, lprobs = self.forward_parser(captions, caption_lengths)
        if inference:
            return argmax_spans
        else:
            ll_loss = nll
            kl_loss = kl
            loss = self.vse_lm_alpha * (ll_loss + kl_loss)

            ReconPPL = torch.sum(ll_loss)/torch.sum(caption_lengths)
            KL = torch.sum(kl_loss)/len(caption_lengths)
            log_PPLBound = torch.sum(ll_loss+kl_loss)/torch.sum(caption_lengths)

            return argmax_spans, loss, ReconPPL, KL, log_PPLBound


class VGCPCFGs(CPCFGs):
    def __init__(self, cfg):
        super(VGCPCFGs, self).__init__(cfg)
        self.img_enc = get_video_encoder(cfg.VIDEO_ENCODER.NAME)(cfg)
        self.txt_enc = get_span_encoder(cfg.SPAN_ENCODER.NAME)(cfg)
        self.vse_mt_alpha = self.cfg.vse_mt_alpha
        self.loss_criterion = getattr(loss, cfg.LOSS.NAME)(cfg)

    def forward_parser(self, captions, caption_lengths):
        params, kl = self.parser(captions, caption_lengths)
        dist = SentCFG(params, lengths=caption_lengths)

        the_spans = dist.argmax[-1]
        argmax_spans, trees, lprobs = extract_parses(the_spans, caption_lengths.tolist(), inc=0)

        ll, span_margs = dist.inside_im
        nll = -ll
        kl = torch.zeros_like(nll) if kl is None else kl
        return nll, kl, span_margs, argmax_spans, trees, lprobs

    def forward_loss(self, base_img_emb, cap_span_features, lengths, span_margs):
        b = base_img_emb.size(0)
        mstep = (lengths * (lengths - 1) // 2).int()
        # focus on only short spans
        if self.cfg.short_span_ratio < 1:
            nstep = max(int(mstep.float().mean().item() * self.cfg.short_span_ratio), 1)
        else:
            nstep = max(self.cfg.short_span_ratio, 1)

        matching_loss_matrix = torch.zeros(b, nstep, device=base_img_emb.device)
        for k in range(nstep):
            img_emb = base_img_emb
            cap_emb = cap_span_features[:, k]

            cap_marg = span_margs[:, k].softmax(-1).unsqueeze(-2)
            cap_emb = torch.matmul(cap_marg, cap_emb).squeeze(-2)
            if "cap_normalize" not in self.cfg or self.cfg.cap_normalize:
                cap_emb = F.normalize(cap_emb, dim=-1)
            loss = self.loss_criterion(img_emb, cap_emb)
            matching_loss_matrix[:, k] = loss
        span_margs = span_margs.sum(-1)
        expected_loss = span_margs[:, : nstep] * matching_loss_matrix
        expected_loss = expected_loss.sum(-1)
        return expected_loss

    def forward(self, inference, tokens, captions, caption_lengths, *videos):
        nll, kl, span_margs, argmax_spans, trees, lprobs = self.forward_parser(captions, caption_lengths)
        if inference:
            return argmax_spans
        else:
            img_emb = self.img_enc(*videos)
            cap_span_features = self.txt_enc(tokens, captions, caption_lengths)
            matching_loss = self.forward_loss(img_emb, cap_span_features, caption_lengths, span_margs)

            mt_loss = matching_loss
            ll_loss = nll
            kl_loss = kl
            loss = self.vse_mt_alpha * mt_loss + self.vse_lm_alpha * (ll_loss + kl_loss)

            ReconPPL = torch.sum(ll_loss)/torch.sum(caption_lengths)
            KL = torch.sum(kl_loss)/len(caption_lengths)
            log_PPLBound = torch.sum(ll_loss+kl_loss)/torch.sum(caption_lengths)
            return argmax_spans, loss, ReconPPL, KL, log_PPLBound

class MMCPCFGs(VGCPCFGs):
    def __init__(self, cfg):
        super(MMCPCFGs, self).__init__(cfg)
        self.img_enc = get_video_encoder(cfg.VIDEO_ENCODER.NAME)(cfg)
        self.loss_criterion = getattr(loss, cfg.LOSS.NAME)(cfg)

    def forward_loss(self, base_img_emb, cap_span_features, lengths, span_margs, return_expert_scores=False):
        b = base_img_emb.size(0)
        mstep = (lengths * (lengths - 1) // 2).int()
        # focus on only short spans
        nstep = int(mstep.float().mean().item() / 2)

        matching_loss_matrix = torch.zeros(b, nstep, device=base_img_emb.device)
        expert_score_matrix = torch.zeros(b, nstep, len(config.DATASET.EXPERTS), device=base_img_emb.device)
        for k in range(nstep):
            img_emb = base_img_emb
            cap_emb = cap_span_features[:, k]

            cap_marg = span_margs[:, k].softmax(-1).unsqueeze(-2)
            cap_emb = torch.matmul(cap_marg, cap_emb).squeeze(-2)
            cap_emb = F.normalize(cap_emb, dim=-1)
            loss, expert_score = self.loss_criterion(img_emb, cap_emb)
            matching_loss_matrix[:, k] = loss
            expert_score_matrix[:,k] = expert_score
        span_margs = span_margs.sum(-1)
        expected_loss = span_margs[:, : nstep] * matching_loss_matrix
        expected_loss = expected_loss.sum(-1)
        if return_expert_scores:
            return expert_score_matrix
        else:
            return expected_loss

    def forward(self, inference, tokens, captions, caption_lengths, *videos):
        return super(MMCPCFGs, self).forward(inference, tokens, captions, caption_lengths, *videos)
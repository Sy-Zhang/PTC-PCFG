import torch
from torch import nn
from .module import GatedEmbedding
from einops import rearrange

class ContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(ContrastiveLoss, self).__init__()
        self.min_val = 1e-8
        self.margin = cfg.LOSS.PARAMS.margin

    def forward(self, vid, txt):
        scores = vid.mm(txt.t()) # cosine similarity
        diagonal = scores.diag().view(vid.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(1)
        loss_img = loss_img.mean(0)
        return loss_txt + loss_img

class MixedContrastiveLoss(ContrastiveLoss):
    def __init__(self, cfg):
        self.cfg = cfg.MODEL.PARAMS
        super(MixedContrastiveLoss, self).__init__(cfg)
        self.gated_emb = GatedEmbedding(cfg)
        self.weight_predictor = nn.Sequential(nn.Linear(self.cfg.sem_dim, len(cfg.DATASET.EXPERTS)), nn.Softmax(dim=-1))

    def forward(self, vid, txt):

        w = self.weight_predictor(txt)
        txt = self.gated_emb(txt)
        scores = torch.sum(w.t()[...,None]*vid.permute(1,0,2).bmm(txt.permute(1,2,0)), dim=0)
        diagonal = scores.diag().view(vid.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)

        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)

        loss_txt = loss_txt.mean(1)
        loss_img = loss_img.mean(0)
        return loss_txt + loss_img, w

class MILMixedContrastiveLoss(MixedContrastiveLoss):
    def __init__(self, cfg):
        super(MILMixedContrastiveLoss, self).__init__(cfg)
        self.mil_type = getattr(cfg, "mil_type", None)

    def forward(self, vid, txt):

        w = self.weight_predictor(txt)
        txt = self.gated_emb(txt)[:,None].repeat(1, vid.shape[1], 1, 1)
        scores = torch.sum(w.repeat(vid.shape[1], 1).t()[...,None]
                           *rearrange(vid, 'b q t d -> t (b q) d')
                           .bmm(rearrange(txt, 'b q t d -> t d (b q)'))
                           , dim=0)
        diagonal = scores.diag().view(scores.size(0), 1)
        if self.mil_type == 'min':
            diagonal = torch.min(diagonal.view(*vid.shape[:2]), dim=1)[0][:,None].repeat(1, vid.shape[1]).view(-1, 1)
        elif self.mil_type == 'max':
            diagonal = torch.max(diagonal.view(*vid.shape[:2]), dim=1)[0][:,None].repeat(1, vid.shape[1]).view(-1, 1)
        elif self.mil_type == 'avg':
            diagonal = torch.mean(diagonal.view(*vid.shape[:2]), dim=1)[:,None].repeat(1, vid.shape[1]).view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_txt = (self.margin + scores - d1).clamp(min=self.min_val)
        loss_img = (self.margin + scores - d2).clamp(min=self.min_val)

        I = torch.eye(scores.size(0))
        if self.mil_type in ['min', 'max', 'avg']:
            for i in range(vid.shape[0]):
                I.view(*vid.shape[:2],*vid.shape[:2])[i, :, i, :] = 1
        I = I  > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_txt = loss_txt.masked_fill_(I, 0)
        loss_img = loss_img.masked_fill_(I, 0)


        loss_txt = loss_txt.mean(1).view(*vid.shape[:2]).mean(-1)
        loss_img = loss_img.mean(0).view(*vid.shape[:2]).mean(-1)
        return loss_txt + loss_img, w
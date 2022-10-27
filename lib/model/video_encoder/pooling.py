import torch
from torch import nn
import torch.nn.functional as F

class AvgPooling(nn.Module):
    def __init__(self, cfg):
        super(AvgPooling, self).__init__()
        self.cfg = cfg.VIDEO_ENCODER.PARAMS

    def forward(self, *videos):
        videos = torch.cat(*videos, dim=-1)
        if self.cfg.normalize:
            videos = F.normalize(videos ,dim=-1)
        videos = torch.mean(videos, dim=1)
        return videos

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.cfg = cfg.VIDEO_ENCODER.PARAMS
        self.normalize = self.cfg.normalize
        in_dim = sum([getattr(self.cfg, '{}_dim'.format(key)) for key in cfg.DATASET.EXPERTS])
        self.fc = nn.Linear(in_dim, cfg.MODEL.PARAMS.sem_dim)

    def forward(self, *images):
        images = torch.cat([torch.mean(im, dim=1) for im in images], dim=-1)
        features = self.fc(images)
        if self.normalize:
            features = F.normalize(features, dim=-1)
        return features



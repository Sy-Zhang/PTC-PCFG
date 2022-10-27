import torch
from torch import nn
import torch.nn.functional as F

class CLIPEncoder(nn.Module):
    def __init__(self, cfg):
        super(CLIPEncoder, self).__init__()

    def forward(self, *videos):
        video_feature = torch.mean(torch.cat(videos, dim=1), dim=1)
        video_feature = F.normalize(video_feature, dim=-1)
        return video_feature
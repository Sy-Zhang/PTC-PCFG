import torch
from torch import nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

class ResNetEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResNetEncoder, self).__init__()
        self.cfg = cfg.VIDEO_ENCODER.PARAMS
        # state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth')
        # self.fc = nn.Linear(1024, 512)
        # self.load_state_dict(state_dict, strict=False)
        # for param in self.fc.parameters():
        #     param.requires_grad = False
        # self.projector = nn.Linear(512, cfg.MODEL.PARAMS.sem_dim)

    def forward(self, *videos):
        videos = torch.mean(torch.cat(*videos, dim=-1), dim=1)
        videos = self.fc(videos)
        videos = self.projector(videos)
        if self.cfg.normalize:
            videos = F.normalize(videos, dim=-1)
        return videos

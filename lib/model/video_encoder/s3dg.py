import torch
from torch import nn
import torch.nn.functional as F

class S3DGEmbedding(nn.Module):
    def __init__(self, cfg):
        super(S3DGEmbedding, self).__init__()
        state_dict = torch.load('.cache/howto100m/s3d_howto100m.pth')
        self.fc = nn.Linear(1024, 512)
        self.load_state_dict(state_dict, strict=False)
        for param in self.fc.parameters():
            param.requires_grad = cfg.VIDEO_ENCODER.PARAMS.finetune

    def forward(self, *videos):
        videos = torch.cat(videos, dim=-1)
        videos = self.fc(videos)
        return videos

class S3DGAvgPooling(nn.Module):
    def __init__(self, cfg):
        super(S3DGAvgPooling, self).__init__()
        self.embedding = S3DGEmbedding(cfg)

    def forward(self, *videos):
        videos = self.embedding(*videos)
        videos = torch.mean(videos, dim=1)
        return videos

class S3DGEncoder(nn.Module):
    def __init__(self, cfg):
        super(S3DGEncoder, self).__init__()
        self.cfg = cfg.VIDEO_ENCODER.PARAMS
        self.embedding = S3DGEmbedding(cfg)
        self.projector = nn.Linear(512, cfg.MODEL.PARAMS.sem_dim)

    def forward(self, *videos):
        videos = self.embedding(*videos)
        videos = torch.mean(torch.cat(*videos, dim=-1), dim=1)
        videos = self.projector(videos)
        if self.cfg.normalize:
            videos = F.normalize(videos, dim=-1)
        return videos

# from transformers import BertConfig
# from transformers.modeling_bert import BertModel
# class S3DGBERT(nn.Module):
#     def __init__(self, cfg):
#         super(S3DGBERT, self).__init__()
#         self.cfg = cfg.VIDEO_ENCODER.PARAMS
#         hidden_size = cfg.MODEL.PARAMS.sem_dim
#         self.embedding = S3DGEmbedding(cfg)
#         bert_config = BertConfig(
#             hidden_size=hidden_size,
#             num_hidden_layers=self.cfg.num_hidden_layers,
#             intermediate_size=self.cfg.intermediate_size,
#             hidden_act=self.cfg.hidden_act,
#             attention_probs_dropout_prob=self.cfg.attention_probs_dropout_prob,
#             num_attention_heads=self.cfg.num_attention_heads,
#             hidden_dropout_prob=self.cfg.hidden_dropout_prob,
#             type_vocab_size=len(cfg.DATASET.EXPERTS),
#         )
#         self.encoder = BertModel(bert_config)
#
#     def forward(self, *videos):
#         videos = self.embedding(*videos)
#         raise NotImplementedError
#         if 'pooling' not in self.cfg:
#             videos = videos[:, 1:]
#         position_ids = torch.arange(0, videos.shape[1])[None].repeat(videos.shape[0], 1).to(videos.device)
#         videos = self.encoder(position_ids=position_ids, inputs_embeds=videos).last_hidden_state
#         if 'pooling' in self.cfg:
#             if self.cfg.pooling == 'avg':
#                 videos = torch.mean(videos, dim=1)
#             else:
#                 videos = torch.max(videos, dim=1)[0]
#         else:
#             videos = videos[:,0]
#         if self.cfg.normalize:
#             videos = F.normalize(videos, dim=-1)
#         return videos
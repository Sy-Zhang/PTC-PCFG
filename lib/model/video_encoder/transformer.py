import torch
from torch import nn
from .position_encoding import build_position_encoding
from .detr_module import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange
import torch.nn.functional as F

class TemporalTransformer(nn.Module):
    def __init__(self, cfg):
        super(TemporalTransformer, self).__init__()
        self.cfg = cfg.VIDEO_ENCODER.PARAMS
        hidden_size = cfg.MODEL.PARAMS.sem_dim

        assert len(cfg.DATASET.EXPERTS) == 1
        self.video_embedding = nn.Linear(self.cfg.get("{}_dim".format(cfg.DATASET.EXPERTS[0])), hidden_size)

        self.position_embedding = build_position_encoding(cfg)

        encoder_layer = TransformerEncoderLayer(hidden_size, self.cfg.nhead, normalize_before=self.cfg.normalize_before)
        encoder_norm = nn.LayerNorm(hidden_size) if self.cfg.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, self.cfg.num_encoder_layers, encoder_norm)


    def forward(self, *videos):
        # videos listed as ['appearance', 'motion', 'audio', 'scene', 'ocr', 'face', 'speech']
        features = self.video_embedding(*videos[0])

        position_embeddings = self.position_embedding(features, torch.zeros(features.shape[:2], dtype=torch.long, device=features.device))
        output = self.encoder(features.permute(1,0,2), pos=position_embeddings.permute(1,0,2))
        if self.cfg.normalize:
            output = F.normalize(output, dim=-1)

        return output[0]

# class TemporalBERT(nn.Module):
#     def __init__(self, cfg):
#         super(TemporalBERT, self).__init__()
#         self.cfg = cfg.VIDEO_ENCODER.PARAMS
#         hidden_size = cfg.MODEL.PARAMS.sem_dim
#
#         assert len(cfg.DATASET.EXPERTS) == 1
#         self.video_embedding = nn.Linear(self.cfg.get("{}_dim".format(cfg.DATASET.EXPERTS[0])), hidden_size)
#
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
#
#         self.encoder = BertModel(bert_config)
#     def forward(self, *videos):
#         # videos listed as ['appearance', 'motion', 'audio', 'scene', 'ocr', 'face', 'speech']
#         features = self.video_embedding(*videos[0])
#         position_ids = torch.arange(0, features.shape[1])[None].repeat(features.shape[0], 1).to(features.device)
#         output = self.encoder(position_ids=position_ids, inputs_embeds=features).last_hidden_state
#         if self.cfg.normalize:
#             output = F.normalize(output, dim=-1)
#         return output[:,0]

class MultiModalTransformer(nn.Module):
    def __init__(self, cfg):
        super(MultiModalTransformer, self).__init__()
        self.cfg = cfg.VIDEO_ENCODER.PARAMS
        hidden_size = cfg.MODEL.PARAMS.sem_dim

        self.video_embeddings = nn.ModuleList()
        for expert in cfg.DATASET.EXPERTS:
            self.video_embeddings.append(nn.Linear(self.cfg.get("{}_dim".format(expert)), hidden_size))

        self.expert_embedding = nn.Embedding(len(cfg.DATASET.EXPERTS), hidden_size)
        self.position_embedding = build_position_encoding(cfg)

        encoder_layer = TransformerEncoderLayer(hidden_size, self.cfg.nhead, normalize_before=self.cfg.normalize_before)
        encoder_norm = nn.LayerNorm(hidden_size) if self.cfg.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, self.cfg.num_encoder_layers, encoder_norm)

    def forward(self, *videos):
        # videos listed as ['appearance', 'motion', 'audio', 'scene', 'ocr', 'face', 'speech']
        videos = [torch.cat([torch.mean(v, dim=1, keepdim=True), v], dim=1) if v.shape[1] > 1 else v for v in videos]

        features = []
        expert_ids = []
        for i, (linear, feat) in enumerate(zip(self.video_embeddings, videos)):
            features.append(linear(feat))
            expert_ids.append(torch.full(feat.shape[:2], i, dtype=torch.long, device=feat.device))
        features = torch.cat(features, dim=1)
        expert_ids = torch.cat(expert_ids, dim=1)
        expert_embeddings = self.expert_embedding(expert_ids)

        position_embeddings = torch.cat([self.position_embedding(
            features, torch.zeros(v.shape[:2], dtype=torch.long, device=features.device)) for v in videos], dim=1)
        output = self.encoder(features.permute(1,0,2), pos=expert_embeddings.permute(1,0,2)+position_embeddings.permute(1,0,2))

        # Handle avg+fixed_seg
        if len(videos) != len(output):
            indexes = torch.cumsum(torch.tensor([0]+[v.shape[1] for v in videos[:-1]], dtype=torch.long, device=output.device), dim=0)
            output = output.index_select(0, indexes)
        output = output.permute(1, 0, 2)
        if self.cfg.normalize:
            output = F.normalize(output, dim=-1)
        return output

# from transformers import BertConfig
# from transformers.modeling_bert import BertModel
# class MultiModalBERT(nn.Module):
#     def __init__(self, cfg):
#         super(MultiModalBERT, self).__init__()
#         self.cfg = cfg.VIDEO_ENCODER.PARAMS
#         self.video_embeddings = nn.ModuleList()
#         hidden_size = cfg.MODEL.PARAMS.sem_dim
#         for expert in cfg.DATASET.EXPERTS:
#             self.video_embeddings.append(nn.Linear(self.cfg.get("{}_dim".format(expert)), hidden_size))
#
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
#
#         self.encoder = BertModel(bert_config)
#
#     def forward(self, *videos):
#         features = []
#         expert_ids = []
#         position_ids = []
#         for i, (linear, feat) in enumerate(zip(self.video_embeddings, videos)):
#             features.append(linear(feat))
#             expert_ids.append(torch.full(feat.shape[:2], i, dtype=torch.long, device=feat.device))
#             position_ids.append(torch.arange(0, feat.shape[1])[None].repeat(feat.shape[0], 1))
#         features = torch.cat(features, dim=1)
#         expert_ids = torch.cat(expert_ids, dim=1)
#         position_ids = torch.cat(position_ids, dim=1).to(expert_ids.device)
#         output = self.encoder(token_type_ids=expert_ids, position_ids=position_ids, inputs_embeds=features).last_hidden_state
#         if len(videos) > 1:
#             indexes = torch.cumsum(torch.tensor([0]+[v.shape[1] for v in videos[:-1]], dtype=torch.long, device=output.device), dim=0)
#             output = output.index_select(1, indexes)
#         if self.cfg.normalize:
#             output = F.normalize(output, dim=-1)
#         return output

class MILMultiModalTransformer(MultiModalTransformer):
    def forward(self, *videos):
        # videos listed as ['appearance', 'motion', 'audio', 'scene', 'ocr', 'face', 'speech']
        features = []
        expert_ids = []
        batch_size = videos[0].shape[0]
        if videos[0].dim() == 4:
            videos = [rearrange(v, "b q t d -> (b q) t d") for v in videos]

        for i, (linear, feat) in enumerate(zip(self.video_embeddings, videos)):
            features.append(linear(feat))
            expert_ids.append(torch.full(feat.shape[:2], i, dtype=torch.long, device=feat.device))
        features = torch.cat(features, dim=1)
        expert_ids = torch.cat(expert_ids, dim=1)
        expert_embeddings = self.expert_embedding(expert_ids)

        position_embeddings = torch.cat([self.position_embedding(
            features, torch.zeros(v.shape[:2], dtype=torch.long, device=features.device)) for v in videos], dim=1)
        output = self.encoder(features.permute(1,0,2), pos=expert_embeddings.permute(1,0,2)+position_embeddings.permute(1,0,2))

        # Handle avg+fixed_seg
        if len(videos) != len(output):
            indexes = torch.cumsum(torch.tensor([0]+[v.shape[1] for v in videos[:-1]], dtype=torch.long, device=output.device), dim=0)
            output = output.index_select(0, indexes)
        output = output.permute(1, 0, 2)
        if batch_size != output.shape[0]:
            output = rearrange(output, "(b q) t d -> b q t d", b=batch_size)
        if self.cfg.normalize:
            output = F.normalize(output, dim=-1)
        return output
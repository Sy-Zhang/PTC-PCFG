from .pooling import ImageEncoder, AvgPooling
from .s3dg import S3DGEncoder, S3DGEmbedding, S3DGAvgPooling
from .clip import CLIPEncoder
from .resnet import ResNetEncoder
from .transformer import MultiModalTransformer, TemporalTransformer

encoder_factory = {
    'CLIPEncoder': CLIPEncoder,
    "TemporalTransformer": TemporalTransformer,
    "S3DGEmbedding": S3DGEmbedding,
    'S3DGAvgPooling': S3DGAvgPooling,
    'S3DGEncoder': S3DGEncoder,
    'ResNetEncoder': ResNetEncoder,
    "AvgPooling": AvgPooling,
    "ImageEncoder": ImageEncoder,
    'MultiModalTransformer': MultiModalTransformer,
}

def get_encoder(name):
    return encoder_factory[name]

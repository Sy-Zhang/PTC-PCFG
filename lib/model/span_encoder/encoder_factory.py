from .lstm_encoder import LSTMEncoder
from .clip_encoder import CLIPEncoder
from .s3dg_encoder import S3DGEncoder
from .bert_encoder import BERTEncoder

encoder_factory = {
    "LSTM": LSTMEncoder,
    'CLIP': CLIPEncoder,
    'S3DG': S3DGEncoder,
    'BERT': BERTEncoder,
}

def get_encoder(name):
    return encoder_factory[name]



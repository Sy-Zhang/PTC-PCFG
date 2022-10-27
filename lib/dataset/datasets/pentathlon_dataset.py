from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle
from nltk import word_tokenize

UNK = 0

class PentathlonDataset(Dataset):
    def __init__(self, cfg, name, split):
        super(PentathlonDataset, self).__init__()
        self.cfg = cfg.DATASET
        self.name = name.split('-')[0]
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.int2word, self.word2int = self.load_vocabularies()
        self.pairs = self.load_annotations()

        if self.split == 'train':
            self.resnext101_features = self.load_segment_features(self.cfg.RESNEXT101_FEATURE_PATH) if 'resnext101' in self.cfg.EXPERTS else None
            self.senet154_features = self.load_segment_features(self.cfg.SENET154_FEATURE_PATH) if 'senet154' in self.cfg.EXPERTS else None
            self.i3d_features = self.load_segment_features(self.cfg.I3D_FEATURE_PATH) if 'i3d' in self.cfg.EXPERTS else None
            self.s3dg_features = self.load_segment_features(self.cfg.S3DG_FEATURE_PATH) if 's3dg' in self.cfg.EXPERTS else None
            self.r2p1d_features = self.load_segment_features(self.cfg.R2P1D_FEATURE_PATH) if 'r2p1d' in self.cfg.EXPERTS else None
            self.densenet161_features = self.load_segment_features(self.cfg.DENSENET161_FEATURE_PATH) if 'densenet161' in self.cfg.EXPERTS else None
            self.ocr_features = self.load_ocr_features() if 'ocr' in self.cfg.EXPERTS else None
            self.face_features = self.load_face_features() if 'face' in self.cfg.EXPERTS else None
            self.audio_features = self.load_audio_features() if 'audio' in self.cfg.EXPERTS else None
            self.speech_features = self.load_speech_features() if 'speech' in self.cfg.EXPERTS else None

    def load_annotations(self):
        pairs = []
        if self.split == 'train':
            subfolder = 'challenge-release-1'
            vids = [l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, self.name, subfolder, 'train_list.txt'))]\
                   +[l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, self.name, subfolder, 'val_list.txt'))]
        elif self.split == 'val':
            subfolder = 'challenge-release-1'
            vids = [l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, self.name, subfolder, 'public_server_val.txt'))]
            sent2tree = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, subfolder, 'non_binary_tree.pkl'), 'rb'))
        else:
            subfolder = 'challenge-release-2'
            vids = [l.rstrip('\n') for l in open(os.path.join(self.cfg.DATA_ROOT, self.name, subfolder, 'public_server_test.txt'))]
            sent2tree = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, subfolder, 'non_binary_tree.pkl'), 'rb'))

        captions = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, subfolder, 'processed-captions.pkl'), 'rb'))

        for vid in vids:
            sentences = captions[vid]
            for tokens in sentences:
                if self.cfg.MAX_TEXT_LENGTH < 0 or len(tokens) < self.cfg.MAX_TEXT_LENGTH or self.split != 'train':
                    sent = ' '.join(tokens)
                    if self.split == 'train':
                        pairs.append({'video_id': vid, 'sentence': sent, 'tokens': tokens})
                    else:
                        tree, span, label = sent2tree[sent]['tree'], sent2tree[sent]['span'], sent2tree[sent]['label']
                        pairs.append({'video_id': vid, 'sentence': sent, 'tree': tree, 'span': span, 'label': label, 'tokens': tokens})
        pairs = sorted(pairs, key=lambda x: len(x['sentence'].split(' ')))[::-1]
        return pairs

    def load_vocabularies(self):
        int2word = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.INT2WORD_PATH), 'rb'))
        word2int = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.cfg.WORD2INT_PATH), 'rb'))
        #add unknown
        int2word = {k+1:v for k, v in int2word.items()}
        word2int = {k:v+1 for k, v in word2int.items()}
        int2word[UNK] = '<unk>'
        word2int['<unk>'] = UNK

        return int2word, word2int


    def load_segment_features(self, path):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, path), 'rb'))
        # if 'fixed_seg' in path:
        #     agg_features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, path.replace('fixed_seg', self.cfg.POOLING_TYPE)), 'rb'))
        #     for k, v in features.items():
        #         features[k] = np.concatenate([agg_features[k], features[k]], axis=0)
        return features

    def load_ocr_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, self.cfg.OCR_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,300), dtype=np.float32)
        return features

    def load_speech_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, self.cfg.SPEECH_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,300), dtype=np.float32)
        return features

    def load_audio_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, self.cfg.AUDIO_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,128), dtype=np.float32)
        return features

    def load_face_features(self):
        features = pickle.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, self.cfg.FACE_FEATURE_PATH), 'rb'))
        for k, v in features.items():
            if isinstance(v, float):
                features[k] = np.zeros((1,512), dtype=np.float32)
        return features

    def process_sentence(self, tokens):
        tokens = torch.tensor([self.word2int.get(word, UNK) for word in tokens], dtype=torch.long)
        return tokens

    def __getitem__(self, index):
        vid = os.path.splitext(self.pairs[index]['video_id'])[0]
        sent = self.pairs[index]['sentence']
        tokens = self.process_sentence(self.pairs[index]['tokens'])

        if self.split in ['train']:
            features = [getattr(self, "{}_features".format(key))[vid] for key in self.cfg.EXPERTS]
            item = {'video_features': features, 'caption': tokens, 'raw_caption': self.pairs[index]['tokens'], 'tree': None, 'span': None, 'label': None}
        else:
            tree = self.pairs[index]['tree']
            span = self.pairs[index]['span']
            label = self.pairs[index]['label']
            item = {'video_features': [], 'caption': tokens, 'tree': tree, 'span': span, 'label': label,
                    'raw_caption': self.pairs[index]['tokens']}

        return item

    def __len__(self):
        return len(self.pairs)
import os
import json
import pickle
import tarfile

import torchtext.vocab
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import io
import math
import numpy as np
from collections import defaultdict
from .utils import load_segment_feature
from model.span_encoder.clip.simple_tokenizer import SimpleTokenizer
UNK = 0


class HowTo100M(Dataset):
    def __init__(self, cfg, name, split, verbose=False, rank=0, world_size=1, num_tars=256):
        super(HowTo100M, self).__init__()
        self.cfg = cfg.DATASET
        self.span_encoder_name = cfg.SPAN_ENCODER.NAME
        self.split = split
        self.name = name
        self.rank = rank
        self.verbose = verbose
        self.world_size = world_size
        self.num_tars = num_tars
        assert self.split in ['train', 'val', 'test']
        if cfg.SPAN_ENCODER.NAME == 'CLIP':
            self.tokenizer = SimpleTokenizer()
        elif cfg.SPAN_ENCODER.NAME == 'S3DG':
            token_to_word = np.load(".cache/howto100m/s3d_dict.npy")
            self.word_to_token = {}
            for i, t in enumerate(token_to_word):
                self.word_to_token[t] = i + 1
        self.int2word, self.word2int = self.load_vocabularies()
        self.pairs = self.load_annotations()
        if 'ocr' in self.cfg.EXPERTS:
            self.vocab  = torchtext.vocab.GloVe(cache='/cache')
            # self.vocab = torchtext.vocab.GloVe(cache='.cache/torchtext')

    def load_annotations(self):
        self.basenames = []
        for a in "0 1 2 3 4 5 6 7 8 9 a b c d e f".split():
            for b in "0 1 2 3 4 5 6 7 8 9 a b c d e f".split():
                self.basenames.append(f"{a}{b}")
        self.basenames = self.basenames[:math.ceil(self.num_tars)]
        if self.num_tars > 1:
            basenames = sorted(self.basenames)
        else:
            basenames = [self.basenames[0]]

        self.feat_paths = defaultdict(dict)

        def load_csv(expert):
            feat_dict = {}
            for i, l in enumerate(open(os.path.join(self.cfg.DATA_ROOT, self.name, f"{expert}.csv")).readlines()):
                video_id, feature_file = l[:-1].split(',')
                feat_dict[video_id] = feature_file
            return feat_dict
        csv_dict = load_csv('s3dg')
        video_ids = json.load(open(os.path.join(self.cfg.DATA_ROOT, self.name, "video_ids.json"), 'r'))

        for video_id in video_ids:
            expert = 's3dg'
            feature_file = csv_dict[expert][video_id]
            basename = feature_file[0]+feature_file[2]
            if basename in self.basenames:
                self.feat_paths[video_id][expert] = feature_file

        self.fn2tarmem = defaultdict(dict)
        annotations = {}
        for basename in tqdm(basenames, dynamic_ncols=True, desc="Loading Tars"):
            pkl_path = os.path.join(self.cfg.DATA_ROOT, self.name, 'processed_captions', f"{basename}.pickle")
            annotation = pickle.load(open(pkl_path, 'rb'))
            annotations.update(annotation)
            for expert in self.cfg.EXPERTS:
                tar_path = os.path.join(self.cfg.DATA_ROOT, "HowTo100M", f"tar.{expert}/{basename}.tar")
                with tarfile.open(tar_path) as tar_fid:
                    tar_members = tar_fid.getmembers()
                    for idx,member in enumerate(tar_members):
                        key = member.name
                        if key[-3:] in ['npy', 'npz', 'pkl']:
                            video_id = os.path.splitext(os.path.basename(key))[0]
                            self.feat_paths[video_id][expert] = key
                            self.fn2tarmem[expert][key] = member

        pairs = []
        for vid, sentences in tqdm(annotations.items(), dynamic_ncols=True, desc='Loading Sentences'):
            if vid not in video_ids:
                continue
            for sent in sentences:
                time, tokens = sent['time'], sent['description']
                if len(tokens) == 0:
                    continue
                if self.cfg.MAX_TEXT_LENGTH < 0 or len(tokens) < self.cfg.MAX_TEXT_LENGTH or self.split != 'train':
                    sent = ' '.join(tokens)
                    assert self.split == 'train'
                    pairs.append({'video_id': vid, 'sentence': sent, 'time': time, 'tokens': tokens})
        if self.num_tars <= 1:
            pairs = pairs[::int(1/self.num_tars)]#[self.rank::self.world_size]
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

    def process_sentence(self, tokens):
        tokens = torch.tensor([self.word2int.get(word, UNK) for word in tokens], dtype=torch.long)
        return tokens

    def get_word_features(self, seg_words, expert_dim):
        if len(seg_words) == 0:
            feature = torch.zeros(expert_dim)
        else:
            feature = torch.mean(torch.stack([self.vocab.vectors[self.vocab.stoi[w]] for w in seg_words], dim=0), dim=0)
        return feature

    def get_tar_video_features(self, expert, video_id):
        if expert == 's3dg':
            key = os.path.join('feature.s3dg', self.feat_paths[video_id][expert])
            basename = self.feat_paths[video_id][expert][0]+self.feat_paths[video_id][expert][2]
            with tarfile.open(os.path.join(self.cfg.DATA_ROOT, self.name, f'tar.{expert}', f'{basename}.tar')) as tar_fid:
                feat = np.load(io.BytesIO(tar_fid.extractfile(self.fn2tarmem[expert][key]).read()))
            features = torch.from_numpy(feat).float()
        elif expert == 'ocr':
            key = self.feat_paths[video_id][expert]
            basename = key[:2]
            if key in self.fn2tarmem[expert]:
                with tarfile.open(os.path.join(self.cfg.DATA_ROOT, self.name, f'tar.{expert}', f'{basename}.tar')) as tar_fid:
                    try:
                        ocr_words = pickle.load(io.BytesIO(tar_fid.extractfile(self.fn2tarmem[expert][key]).read()))
                    except:
                        print(key)
                        raise NotImplementedError
                features = {}
                for k, v in ocr_words.items():
                    seg_words = [w for w in v if w in self.vocab.stoi]
                    features[k] = self.get_word_features(seg_words, 300)

        else:
            key = self.feat_paths[video_id][expert]
            basename = key[:2]
            with tarfile.open(os.path.join(self.cfg.DATA_ROOT, self.name, f'tar.{expert}', f'{basename}.tar')) as tar_fid:
                feat = np.load(io.BytesIO(tar_fid.extractfile(self.fn2tarmem[expert][key]).read()))['features']
            features = torch.from_numpy(feat).float()
        return features

    def __getitem__(self, index):
        vid = os.path.splitext(self.pairs[index]['video_id'])[0]
        sent = self.pairs[index]['sentence']
        time = self.pairs[index]['time']
        tokens = self.process_sentence(self.pairs[index]['tokens'])

        features = []

        for expert in self.cfg.EXPERTS:
            time_unit = self.cfg.TIME_UNITS[expert]
            video_feature = self.get_tar_video_features(expert, vid)
            num_clips = self.cfg.NUM_OUTPUT_CLIPS
            if self.cfg.NUM_INSTANCE > 1:
                random_c = np.random.normal((time[1] + time[0])/2, self.cfg.SIGMA_C, self.cfg.NUM_INSTANCE)
                random_w = (time[1] - time[0])*np.exp(np.random.normal(0, self.cfg.SIGMA_W, self.cfg.NUM_INSTANCE))
                random_times = [[r_c-r_w/2, r_c+r_w/2] for r_c, r_w in zip(random_c, random_w)]
                positive_features = torch.stack([load_segment_feature(video_feature, t, num_clips, time_unit) for t in random_times], dim=0)
                features.append(positive_features)
            else:
                features.append(load_segment_feature(video_feature, time, num_clips, time_unit))

        if self.split == 'train':
            item = {'video_features': features, 'caption': tokens, 'tree': None, 'span': None, 'label': None,
                    'raw_caption': self.pairs[index]['tokens']}
        else:
            tree = self.pairs[index]['tree']
            span = self.pairs[index]['span']
            label = self.pairs[index]['label']
            item = {'video_features': features, 'caption': tokens, 'tree': tree, 'span': span, 'label': label,
                    'raw_caption': self.pairs[index]['tokens']}

        return item


    def __len__(self):
        return len(self.pairs)
import torch
import numpy as np

def feature_temporal_sampling(num_samples, features):
    num_clips = features.shape[0]
    idxs = torch.arange(0, num_samples + 1, 1.0) / num_samples * num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_samples):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        new_visual_input.append(features[(s_idx+e_idx)//2])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

def load_segment_feature(video_feature, time, num_clips, time_unit):
    start_idx, end_idx = max(int(np.floor(time[0] / time_unit)),0), int(np.ceil(time[1] / time_unit))
    if type(video_feature) is dict:
        clip_feature = torch.stack([video_feature.get(i, torch.zeros(300)) for i in range(start_idx, end_idx)], dim=0)
    else:
        clip_feature = video_feature[start_idx:end_idx]
    if len(clip_feature) > 0:
        clip_feature = feature_temporal_sampling(num_clips, clip_feature)
    else:
        clip_feature = torch.zeros(num_clips, clip_feature.shape[-1]).float()
    return clip_feature

def load_aggregated_feature(video_feature, pooling_type='avg'):
    if pooling_type == 'avg':
        return torch.mean(video_feature, dim=0, keepdim=True)
    elif pooling_type == 'max':
        return torch.max(video_feature, dim=0, keepdim=True)[0]
    else:
        raise NotImplementedError

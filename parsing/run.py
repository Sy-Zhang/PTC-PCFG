import os
import argparse
import pprint
import pickle
import builtins

from tqdm import tqdm
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import random
import time

import _init_path
from core.utils import all_gather
from core.config import update_config, cfg
from core.utils import create_logger, AverageMeter
from dataset.dataset_factory import get_dataset
from dataset.sampler import DistributedTarSampler, DistributedSampler
from model.model_factory import get_model
from eval import get_batch_stats, initialize_stats, update_stats, get_f1s, display_f1s
from nltk.tokenize.treebank import TreebankWordDetokenizer
global logger

detokenizer = TreebankWordDetokenizer()

def set_environment_variables_for_nccl_backend(verbose=True):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["MASTER_ADDR"] = os.environ["MASTER_IP"] if "MASTER_IP" in os.environ else os.environ["AZ_BATCHAI_JOB_MASTER_NODE_IP"]
        os.environ["MASTER_PORT"] = os.environ["MASTER_PORT"] if "MASTER_PORT" in os.environ else "23456"
        os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
        if verbose:
            print("RANK = {}".format(os.environ["RANK"]))
            print("WORLD_SIZE = {}".format(os.environ["WORLD_SIZE"]))
            print("MASTER_ADDR = {}".format(os.environ["MASTER_ADDR"]))
            print("MASTER_PORT = {}".format(os.environ["MASTER_PORT"]))
            print(
                "NCCL_SOCKET_IFNAME new value = {}".format(os.environ["NCCL_SOCKET_IFNAME"])
            )

def get_dist_url():
    if "MASTER_IP" in os.environ:
        return "tcp://"+ os.environ['MASTER_IP'] + ":" + os.environ['MASTER_PORT']
    else:
        return "env://"

def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    # parallel configs:
    parser.add_argument('--world-size', type=int, help='number of nodes for distributed training',
                        default=os.environ["WORLD_SIZE"])
    parser.add_argument('--rank', type=int, help='node rank for distributed training',
                        default=os.environ["RANK"])
    parser.add_argument('--dist-url', type=str, help='url used to set up distributed training',
                        default=get_dist_url())
    parser.add_argument('--local_rank', type=int, help='local rank/gpu id for distributed training',
                        default=int(os.environ["RANK"]) % torch.cuda.device_count())
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--workers', type=int, default=None, help='number of workers')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    # training
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--no_save', default=False, action="store_true", help='don\'t save checkpoint')
    parser.add_argument('--data_root', help='data path', type=str)
    parser.add_argument('--model_dir', help='model path', type=str)
    parser.add_argument('--log_dir', help='log path', type=str)
    parser.add_argument('--num_tars', type=int, help='number of tars', default=-1)
    parser.add_argument('--max_epoch', type=int, help='max epoch')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--resume', default=False, action="store_true", help='train with punctuation')
    # testing
    parser.add_argument('--checkpoint', help='checkpoint path', type=str)
    parser.add_argument('--display_by_length', default=False, action="store_true", help='don\'t save checkpoint')
    parser.add_argument('--save_results', default=False, action="store_true", help='save predictions')
    parser.add_argument('--test_mode', default=False, action="store_true", help='run test epoch only')
    parser.add_argument('--split', help='test split', type=str, default='test')
    args = parser.parse_args()

    cfg.RESUME = args.resume
    if args.data_root:
        cfg.DATASET.DATA_ROOT = args.data_root
    if args.num_tars > 0:
        cfg.DATASET.NUM_TARS = args.num_tars
    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir
    if args.log_dir:
        cfg.LOG_DIR = args.log_dir
    if args.checkpoint:
        cfg.CHECKPOINT = args.checkpoint
    if args.max_epoch:
        cfg.OPTIM.MAX_EPOCH = args.max_epoch
    if args.workers is not None:
        cfg.DATALOADER.WORKERS = args.workers

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.autograd.set_detect_anomaly(True)

    return args

def collate_fn(batch):
    video_features = []
    for feats in zip(*[b['video_features'] for b in batch]):
        video_features.append(torch.from_numpy(np.stack(feats, axis=0)).float())
    captions = pad_sequence([b['caption'] for b in batch], batch_first=True)
    caption_lengths = torch.tensor([b['caption'].shape[0] for b in batch], dtype=torch.long)
    spans = [b['span'] for b in batch]
    labels = [b['label'] for b in batch]
    raw_captions = [b['raw_caption'] for b in batch]
    return spans, labels, raw_captions, captions, caption_lengths, video_features

def network(batch, model, optimizer=None):
    gold_spans, labels, raw_captions, captions, caption_lengths, video_features = batch
    captions = captions.cuda(args.local_rank, non_blocking=True)
    caption_lengths = caption_lengths.cuda(args.local_rank, non_blocking=True)
    video_features = [feat.cuda(args.local_rank, non_blocking=True) for feat in video_features]
    if model.training:
        argmax_spans, loss, ReconPPL, KL, log_PPLBound = model(False, raw_captions, captions, caption_lengths, *video_features)

        loss_value = loss.mean()
        ReconPPL = ReconPPL.mean()
        KL = KL.mean()
        log_PPLBound = log_PPLBound.mean()
        optimizer.zero_grad()
        loss_value.backward()
        if cfg.OPTIM.GRAD_CLIP > 0:
            clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP)
        optimizer.step()
        return loss_value.item(), ReconPPL.item(), KL.item(), log_PPLBound.item()
    else:
        argmax_spans = model(True, raw_captions, captions, caption_lengths)
        pred_spans = [[[a[0], a[1]] for a in span] for span in argmax_spans]
        stats = get_batch_stats(caption_lengths.tolist(), pred_spans, gold_spans, labels)
        predictions = [{'caption': detokenizer.detokenize(tokens), 'span': span} for tokens, span in zip(raw_captions, argmax_spans)]
        return stats, predictions

def run_epoch(data_loader, model, optimizer=None):
    dataset_name = data_loader.dataset.name
    if args.print:
        data_loader = tqdm(data_loader, dynamic_ncols=True)

    loss_meter = AverageMeter()
    ReconPPL_meter = AverageMeter()
    KL_meter = AverageMeter()
    log_PPLBound_meter = AverageMeter()
    all_stats = initialize_stats()
    predictions = []
    for batch in data_loader:
        caption_lengths = batch[4]

        if model.training:
            loss, ReconPPL, KL, log_PPLBound = network(batch, model, optimizer)
            loss_meter.update(loss, 1)
            ReconPPL_meter.update(ReconPPL, torch.sum(caption_lengths).item())
            KL_meter.update(KL, len(caption_lengths))
            log_PPLBound_meter.update(log_PPLBound, torch.sum(caption_lengths).item())
        else:
            mini_batch_size = len(caption_lengths)
            if max(caption_lengths) > 20:
                    mini_batch_size = 2
            for i in range(len(caption_lengths) // mini_batch_size):
                mini_batch = [batch[0]] + [b[i * mini_batch_size:(i + 1) * mini_batch_size] for b in batch[1:]]
                with torch.no_grad():
                    stats, pred_spans = network(mini_batch, model, optimizer)
                predictions.extend(pred_spans)
                update_stats(all_stats, stats)

    if model.training:
        gathered_loss_meter = AverageMeter()
        gathered_ReconPPL_meter = AverageMeter()
        gathered_KL_meter = AverageMeter()
        gathered_log_PPLBound_meter = AverageMeter()
        for var_per_gpu in all_gather(loss_meter):
            gathered_loss_meter.update(var_per_gpu.val, var_per_gpu.count)
        for var_per_gpu in all_gather(ReconPPL_meter):
            gathered_ReconPPL_meter.update(var_per_gpu.val, var_per_gpu.count)
        for var_per_gpu in all_gather(KL_meter):
            gathered_KL_meter.update(var_per_gpu.val, var_per_gpu.count)
        for var_per_gpu in all_gather(log_PPLBound_meter):
            gathered_log_PPLBound_meter.update(var_per_gpu.val, var_per_gpu.count)

        info = {'loss': gathered_loss_meter.avg, 'ReconPPL': gathered_ReconPPL_meter.avg,
                'KL': gathered_KL_meter.avg, 'log_PPLBound': gathered_log_PPLBound_meter.avg}

    else:
        gathered_all_stats = initialize_stats()
        for var_per_gpu in all_gather(all_stats):
            update_stats(gathered_all_stats, var_per_gpu)
        info = {'f1s': get_f1s(gathered_all_stats)}


    return info, predictions

@torch.no_grad()
def test(cfg):
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # suppress printing if not master
    args.print = args.rank == 0 and args.verbose
    if args.rank!=0:
        args.verbose = False
        def print_pass(*args):
            pass
        builtins.print = print_pass

    test_datasets = [get_dataset(name)(cfg, name, args.split) for name in cfg.TEST.DATASET]
    test_loaders = [DataLoader(dataset,
                               batch_size=cfg.DATALOADER.INFERENCE_BATCH_SIZE//args.world_size,
                               num_workers=cfg.DATALOADER.WORKERS,
                               pin_memory=True,
                               sampler=DistributedSampler(dataset, shuffle=False),
                               collate_fn=collate_fn)
                    for dataset in test_datasets]

    if cfg.MODEL.NAME == 'VGCPCFGs_S3DG_Vocab':
        cfg.MODEL.PARAMS.vocabulary_size = 1+len(np.load(".cache/howto100m/s3d_dict.npy"))
    else:
        cfg.MODEL.PARAMS.vocabulary_size = 1+len(pickle.load(open(os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.WORD2INT_PATH), 'rb')))
    model = get_model(cfg.MODEL.NAME)(cfg)
    if cfg.MODEL.NAME not in ['Random', 'LeftBranching', 'RightBranching']:
        assert os.path.exists(cfg.CHECKPOINT), "checkpoint not exists"
        checkpoint = torch.load(cfg.CHECKPOINT)
        model.load_state_dict(checkpoint['model'])
        torch.cuda.set_device(args.local_rank)
        model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])#, find_unused_parameters=True

    model.eval()
    for test_loader in test_loaders:
        with torch.no_grad():
            test_info, predictions = run_epoch(test_loader, model)

        result = display_f1s(test_info['f1s'], f'performance on {test_loader.dataset.name} {args.split} set', args.display_by_length)
        print(result)
        if args.save_results:
            cfg_filename = os.path.splitext(os.path.basename(args.cfg))[0]
            saved_result_filename = os.path.join(cfg.RESULT_DIR, '{}/{}/{}-{}.pkl'.format(
                test_loader.dataset.name, cfg_filename, os.path.splitext(os.path.basename(args.checkpoint))[0], args.split))

            rootfolder = os.path.dirname(saved_result_filename)
            if not os.path.exists(rootfolder):
                print('Make directory %s ...' % rootfolder)
                os.makedirs(rootfolder)

            pickle.dump(predictions, open(saved_result_filename, 'wb'))


def train(cfg):
    global logger
    # args.print = True
    args.print = args.rank == 0 and args.verbose
    if args.rank!=0:
        args.verbose = False
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.print:
        logger, final_output_dir, log_filename = create_logger(cfg, args.cfg, args.tag)
        logger.info('\n'+pprint.pformat(args))
        logger.info('\n' + pprint.pformat(cfg))

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    if cfg.TRAIN.DATASET == 'HowTo100M':
        train_dataset = get_dataset(cfg.TRAIN.DATASET)(cfg, cfg.TRAIN.DATASET, 'train',
                                                       args.print, args.rank, args.world_size, cfg.DATASET.NUM_TARS)
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.DATALOADER.BATCH_SIZE//args.world_size,
                                  num_workers=cfg.DATALOADER.WORKERS,
                                  pin_memory=True,
                                  sampler=DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                             rank=args.rank, shuffle=cfg.TRAIN.SHUFFLE),
                                  # DistributedTarSampler(train_dataset, num_replicas=args.world_size,
                                  #                               shuffle=cfg.TRAIN.SHUFFLE, drop_last=True),
                                  collate_fn=collate_fn)

    else:
        train_dataset = get_dataset(cfg.TRAIN.DATASET)(cfg, cfg.TRAIN.DATASET, 'train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.DATALOADER.BATCH_SIZE//args.world_size,
                                  num_workers=cfg.DATALOADER.WORKERS,
                                  pin_memory=True,
                                  sampler=DistributedSampler(train_dataset, shuffle=cfg.TRAIN.SHUFFLE),
                                  collate_fn=collate_fn)

    test_datasets = [get_dataset(name)(cfg, name, args.split)
                     for name in cfg.TEST.DATASET]
    test_loaders = [DataLoader(dataset,
                               batch_size=cfg.DATALOADER.INFERENCE_BATCH_SIZE//args.world_size,
                               num_workers=cfg.DATALOADER.WORKERS,
                               pin_memory=True,
                               sampler=DistributedSampler(dataset, shuffle=False),
                               collate_fn=collate_fn)
                    for dataset in test_datasets]


    cfg.MODEL.PARAMS.vocabulary_size = len(train_dataset.word2int)
    model = get_model(cfg.MODEL.NAME)(cfg)
    if os.path.exists(cfg.CHECKPOINT) and cfg.RESUME:
        logger.info("load model from {}".format(cfg.CHECKPOINT))
        checkpoint = torch.load(cfg.CHECKPOINT)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
    else:
        start_epoch = 0
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)#

    optimizer = getattr(optim, cfg.OPTIM.NAME)(model.parameters(), lr=cfg.OPTIM.LEARNING_RATE, betas=(cfg.OPTIM.BETA1, 0.999))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        np.random.seed(cur_epoch)
        random.seed(cur_epoch)
        train_loader.sampler.set_epoch(cur_epoch)

        optim_msg, result_msg = '', ''
        model.train()
        train_info, _ = run_epoch(train_loader, model, optimizer)
        optim_msg += "\nepoch: {} lr: {:.6f} ".format(cur_epoch, optimizer.param_groups[0]['lr'])
        train_display = 'Train [ReConPPL: {:.2f}, KL: {:.2f}, PPLBound: {:.2f}]'.format(
            train_info['ReconPPL'], train_info['KL'], np.exp(train_info['log_PPLBound']))
        result_msg += train_display+'\n'
        model.eval()
        for name, loader in zip(cfg.TEST.DATASET, test_loaders):
            with torch.no_grad():
                test_info, _ = run_epoch(loader, model)
                test_display = display_f1s(test_info['f1s'], f'{name} {args.split} Performance', args.display_by_length)
                result_msg += test_display+'\n'
        if args.print:
            logger.info(optim_msg+result_msg+'\n')

        if not args.no_save and args.print:
            cfg_filename = os.path.splitext(os.path.basename(args.cfg))[0]
            saved_model_filename = os.path.join(cfg.MODEL_DIR, '{}/{}/seed-{}/epoch{:04d}.pkl'.format(
                cfg.TRAIN.DATASET, cfg_filename, args.seed, cur_epoch))

            rootfolder = os.path.dirname(saved_model_filename)
            if not os.path.exists(rootfolder):
                print('Make directory %s ...' % rootfolder)
                os.makedirs(rootfolder)

            torch.save({'epoch': cur_epoch, 'model': model.module.state_dict()}, saved_model_filename)


if __name__ == '__main__':
    args = parse_args()
    if args.test_mode:
        test(cfg)
    else:
        train(cfg)
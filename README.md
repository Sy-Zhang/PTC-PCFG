# Learning a Grammar Inducer by Watching Millions of Instructional YouTube Videos

Video-aided grammar induction aims to leverage video information for finding more accurate syntactic grammars for accompanying
text. While [previous work](https://github.com/Sy-Zhang/MMC-PCFG) focuses on building systems on well-aligned video-text pairs, we train our model only on noisy YouTube videos without finetuning on benchmark data and achieved stronger performances across three benchmarks.

[arXiv preprint](https://arxiv.org/pdf/2210.12309.pdf)

## News
- :sunny: Our paper was accepted by EMNLP 2022.

## Approach

![Our framework](figures/framework.png)

## Requirements
We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.

## Quick Start

# Launch Docker Container
```bash
CUDA_VISIBLE_DEVICES=0,1 source launch_container.sh $PATH_TO_STORAGE/data $PATH_TO_STORAGE/checkpoints $PATH_TO_STORAGE/log
```
The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
Note that the source code is mounted into the container under `/src` instead 
of built into the image so that user modification will be reflected without
re-building the image.

# Data Preparation
Please download the data from [here](https://www.dropbox.com/sh/flshflx1rdrxh8v/AAAktPEWL1iHde0wU20aVlwGa?dl=0) to `data`, 
 and [here](https://www.dropbox.com/sh/mjha0m8onhkerxm/AADZmVbXWRVwwg9__f6O98sYa?dl=0) to `.cache`.
Preprocessing details are described [here](#preprocessing-details). 

# Training

Run the following commands for training:
```bash
sh scripts/train.sh
```

# Evaluation
Our trained model are provided [here](https://www.dropbox.com/sh/jjp48bmr8tj283e/AAArbHSQsZQzbNR_TCKN8QIga?dl=0). Please download them to `checkpoints`.
Then, run the following commands for evaluation:
```bash
sh scripts/test.sh
```
## Preprocessing details
We preprocess subtitles with the following scripts:
```bash
python tools/preprocess_captions.py
python tools/compute_gold_trees.py
python tools/generate_vocabularies.py
```

## Acknowledgements

This repo is developed based on [VPCFG](https://github.com/zhaoyanpeng/vpcfg), [MMC-PCFG](https://github.com/Sy-Zhang/MMC-PCFG) and [Punctuator2](https://github.com/ottokart/punctuator2).


## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@inproceedings{zhang2022training,
title={Learning a Grammar Inducer by Watching Millions of Instructional YouTube Videos},
author={Zhang, Songyang and Song, Linfeng and Jin, Lifeng and Mi, Haitao and Xu, Kun and Yu, Dong and Luo, Jiebo},
booktitle={EMNLP},
year={2022}
```

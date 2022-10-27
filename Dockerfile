FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 && \
    apt-get install -y fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN conda install pip==20.1.1
RUN conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
RUN conda install pyyaml nltk tqdm
RUN pip install terminaltables einops easydict
RUN pip install git+https://github.com/zhaoyanpeng/pytorch-struct.git@infer_pos_tag
RUN python -c "import nltk; nltk.download('punkt')"
RUN pip install transformers==3.4
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git
RUN mkdir /cache
RUN wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O /cache/glove.840B.300d.zip && cd /cache/ && unzip glove.840B.300d.zip
RUN pip install torchtext==0.6
RUN pip install boto3
RUN pip install omegaconf
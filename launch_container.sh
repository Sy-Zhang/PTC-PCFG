DATA_DIR=$1
CHECKPOINT_DIR=$2
LOG_DIR=$3

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --network=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$DATA_DIR,dst=/src/data,type=bind,readonly \
    --mount src=$CHECKPOINT_DIR,dst=/src/checkpoints,type=bind \
    --mount src=$LOG_DIR,dst=/src/log,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src zhangsongyang/cpcfg:v1.5.0_cu102
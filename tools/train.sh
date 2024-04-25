work_dir="/home/pod/project/UV-Mamba"
export PYTHONPATH=$work_dir
MASTER_PORT=32990

#CONFIG_FILE="config/uv/segformer/segformer_beijing.yaml"

CONFIG_FILE="config/uv/segformer/segformer_shenzhen.yaml"

#CONFIG_FILE="config/uv/segformer/segformer_xian.yaml"


CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE


#CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT scripts/test_semantic.py $CONFIG_FILE

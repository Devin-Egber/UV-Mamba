work_dir="/data/project/UV-Mamba"
export PYTHONPATH=$work_dir
MASTER_PORT=32990

# Beijing
CONFIG_FILE="config/uv/uvmamba_beijing.yaml"

#xian
CONFIG_FILE="config/uv/uvmamba_xian.yaml"

CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE --fine_tune


work_dir="/data/mamba/Ablation"
export PYTHONPATH=$work_dir
MASTER_PORT=32990


CONFIG_FILE="config/uv/uvmamba_xian.yaml"



CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/train.py --config_file $CONFIG_FILE --fine_tune


#CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT scripts/test_semantic.py $CONFIG_FILE

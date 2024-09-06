work_dir="/data/mamba/Ablation"
export PYTHONPATH=$work_dir


CONFIG_FILE="config/uv/uvmamba_beijing.yaml"


CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 32996 tools/get_flops.py --config_file $CONFIG_FILE



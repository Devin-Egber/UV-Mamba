work_dir="/data/project/UV-Mamba"
export PYTHONPATH=$work_dir


CONFIG_FILE="config/uv/uv_mamba/deform_uvmamba_cityscapes.yaml"


CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 32996 scripts/get_flops.py --config_file $CONFIG_FILE



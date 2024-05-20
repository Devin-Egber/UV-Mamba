work_dir="/data/project/UV-Mamba"
export PYTHONPATH=$work_dir

MASTER_PORT=32990


#CONFIG_FILE="config/uv/segformer/uvmamba_beijing.yaml"

# CONFIG_FILE="config/uv/segformer/segformer_shenzhen.yaml"

#CONFIG_FILE="config/uv/segformer/segmamba_xian.yaml"
# CONFIG_FILE="config/uv/segmamba/segmamba_beijing.yaml"
# CONFIG_FILE="config/uv/segmamba/segmamba_cityspace.yaml"

CONFIG_FILE="config/uv/segmamba/segmamba_beijing.yaml"


# test one of five fold script
weight_folder="weights/global_step7654"
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/test.py --config_file $CONFIG_FILE --weight_folder $weight_folder

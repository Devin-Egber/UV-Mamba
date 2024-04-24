work_dir="/home/pod/project/UV-Mamba"
export PYTHONPATH=$work_dir

MASTER_PORT=32990


CONFIG_FILE="config/uv/segformer/segformer.yaml"


# test one of five fold script
weight_folder="weights/global_step8600"
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $MASTER_PORT tools/test.py --config_file $CONFIG_FILE --weight_folder $weight_folder

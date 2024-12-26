#!/bin/bash

source ../../switch_cuda.sh 11.8

export HF_ENDPOINT=https://hf-mirror.com

config_file="/home/ywt/lab/lora-scripts/a31_char_new copy.toml"	
# config_file2="/home/ywt/lab/lora-scripts/a31_char_new2.toml"	 # config file | 使用 toml 文件指定训练参数


# DIR="/mnt/e/data/genshin_1022数据集/genshin"

# # 遍历DIR中的每个子文件夹
# for folder in "$DIR"/*; do
#   if [ -d "$folder" ]; then  # 确保是文件夹
#     echo "Processing folder: $folder"
    
#     # 调用Python脚本，传入当前子文件夹路径作为 --data_dir 参数
#     python "/home/ywt/lab/sd-scripts/edit_toml.py" --data_dir="$folder" --toml_file="$config_file" --pretrained_model_name_or_path="/mnt/d/sdwebui_model/Stable-diffusion/netaArtXL_v10.safetensors" --end="_neta"
    
#     # 开始训练，使用更新后的config文件
#     python "/home/ywt/lab/sd-scripts/sdxl_train_network.py" --config_file="$config_file"
#   fi
# done


python "/home/ywt/lab/sd-scripts/edit_toml.py" --data_dir="/mnt/e/data/train_nieta/jellycat/" --toml_file="$config_file" --pretrained_model_name_or_path="/mnt/d/sdwebui_model/Stable-diffusion/checkpoint-e3_s178000.safetensors" --end="_noob1"
python "/home/ywt/lab/sd-scripts/sdxl_train_network.py" --config_file="$config_file" \


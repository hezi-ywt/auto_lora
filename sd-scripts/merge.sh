python /home/ywt/lab/sd-scripts/networks/merge.py \
/home/ywt/lab/stable-diffusion-webui/models/Stable-diffusion/kohaku-xl-epsilon-rev1.safetensors  \
/home/ywt/lab/stable-diffusion-webui/models/Lora/noob-lyco9-000001.safetensors \
/home/ywt/lab/stable-diffusion-webui/models/Stable-diffusion/noob-lyco9-000001.safetensors \
--is_sdxl --device cpu --dtype fp16 --weight 1.0
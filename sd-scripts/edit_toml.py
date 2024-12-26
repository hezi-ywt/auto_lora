import os
import argparse
import toml

def edit_toml(
    toml_file="/home/ywt/lab/lora-scripts/a31_char_new.toml",
    pretrained_model_name_or_path=None,
    data_dir=None,
    end="_lora",
    learning_rate=0.0001,
    max_train_epochs=100,
    network_dim=8
):
    with open(toml_file, "r") as f:
        config = toml.load(f)
        
        # 只在提供时更新路径配置
        if data_dir:
            config["Basics"]["train_data_dir"] = data_dir
            config["Save"]["output_name"] = os.path.basename(data_dir) + end
            
        if pretrained_model_name_or_path:
            config["Basics"]["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        
        # 更新其他参数（使用默认值）
        config["Optimizer"]["unet_lr"] = float(learning_rate)
        config["Basics"]["max_train_epochs"] = int(max_train_epochs)
        config["Network_setup"]["network_dim"] = int(network_dim)
        config["Network_setup"]["network_alpha"] = int(network_dim) // 2
        
    with open(toml_file, "w") as f:
        toml.dump(config, f)
    
    print(f"Updated config:")
    if data_dir:
        print(f"- Data dir: {data_dir}")
    if pretrained_model_name_or_path:
        print(f"- Model path: {pretrained_model_name_or_path}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Max train epochs: {max_train_epochs}")
    print(f"- Network dim: {network_dim}")
    print(f"- Network alpha: {network_dim // 2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--toml_file', type=str, default="lora-scripts/a31_char_new.toml", help='config路径')
    parser.add_argument('--data_dir', type=str, help='数据集路径')
    parser.add_argument('--pretrained_model_name_or_path', default="/home/ywt/lab/stable-diffusion-webui/models/Stable-diffusion/noob7.safetensors", type=str, help='预训练模型路径')
    parser.add_argument('--end', type=str, default="_noob7", help='输出模型后缀')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--max_train_epochs', type=int, help='最大训练epoch数')
    parser.add_argument('--network_dim', type=int, help='网络维度')
    
    args = parser.parse_args()

    edit_toml(
        data_dir=args.data_dir,
        toml_file=args.toml_file,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        end=args.end,
        learning_rate=args.learning_rate,
        max_train_epochs=args.max_train_epochs,
        network_dim=args.network_dim
    )

import os
import toml

def edit_toml(data_dir, toml_file, pretrained_model_name_or_path, end="_lora", learning_rate=0.0001, max_train_epochs=20, network_dim=8):
    """编辑 TOML 配置文件"""
    try:
        # 获取数据目录的绝对路径
        abs_data_dir = os.path.abspath(data_dir)
        print(f"Using absolute data dir: {abs_data_dir}")
        
        # 读取 TOML 文件
        with open(toml_file, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        
        # 更新配置
        config['Basics']['train_data_dir'] = abs_data_dir
        config['Basics']['pretrained_model_name_or_path'] = pretrained_model_name_or_path
        config['Save']['output_name'] = end
        config['Optimizer']['unet_lr'] = learning_rate
        config['Basics']['max_train_epochs'] = max_train_epochs
        config['Network_setup']['network_dim'] = network_dim
        
        # 保存修改后的配置
        with open(toml_file, 'w', encoding='utf-8') as f:
            toml.dump(config, f)
            
        print(f"Updated config file {toml_file} with data_dir: {abs_data_dir}")
        
    except Exception as e:
        print(f"Error editing TOML file: {str(e)}")
        raise e 
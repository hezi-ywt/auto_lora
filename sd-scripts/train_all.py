import os
import sys
import argparse
import subprocess
import json
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=str, required=True)
    parser.add_argument('--config_dir', type=str, required=True)
    args = parser.parse_args()
    
    # 读取任务信息
    task_file = os.path.join(args.config_dir, f"{args.task_id}.json")
    with open(task_file, 'r') as f:
        task = json.load(f)
    
    # 获取脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "sdxl_train_network.py")
    
    total = len(task['config_files'])
    for idx, config_file in enumerate(task['config_files'], 1):
        try:
            # 运行训练
            process = subprocess.run([
                "python",
                train_script,  # 使用完整路径
                f"--config_file={config_file}"
            ], check=True)
            
            # 删除已完成的配置文件
            os.remove(config_file)
            
            # 更新进度
            progress = (idx / total) * 100
            update_progress(args.task_id, progress, config_file)
            
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {config_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {config_file}: {e}")
            continue

def update_progress(task_id, progress, current_file):
    """更新进度文件"""
    progress_file = f"progress_{task_id}.json"
    with open(progress_file, 'w') as f:
        json.dump({
            "progress": progress,
            "current_file": current_file,
            "timestamp": time.time()
        }, f)

if __name__ == "__main__":
    main() 
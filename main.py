# 1. 先导入所有需要的模块
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import shutil
import json
from datetime import datetime
import time
import yaml
from fastapi.responses import FileResponse, StreamingResponse
import asyncio
from pathlib import Path
import zipfile
import rarfile
import py7zr
import tempfile
from starlette.requests import Request
from starlette.responses import Response
from collections import deque
import base64
import httpx
import re
import subprocess
import sys
from starlette.websockets import WebSocketDisconnect

# 2. 读取配置文件
def load_config(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.yaml'):
            return yaml.safe_load(f)
        elif file_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format")

# 3. 加载配置
config = load_config("config.yaml")

# 4. 添加 sd-scripts 路径到 Python 路径
sys.path.append(os.path.abspath(config["custom_sd_scripts_dir"]))

# 5. 导入其他模块
from tagger.auto_tag import traverse_tag_images
from edit_toml import edit_toml  # 直接从当前目录导入

# 6. 设置路径变量
CUSTOM_STATIC_DIR = config["custom_static_dir"]
CUSTOM_UPLOAD_DIR = config["custom_upload_dir"]
CUSTOM_ANNOTATION_DIR = config["custom_annotation_dir"]

# 7. 确保目录存在
os.makedirs(CUSTOM_STATIC_DIR, exist_ok=True)
os.makedirs(CUSTOM_UPLOAD_DIR, exist_ok=True)
os.makedirs(CUSTOM_ANNOTATION_DIR, exist_ok=True)

# 8. 创建 FastAPI 应用
app = FastAPI(title="图像生成模型训练平台")

# 9. 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 10. 配置静态文件服务
app.mount("/static", StaticFiles(directory=CUSTOM_STATIC_DIR), name="static")

# 数据标注相关路由
@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile]):
    """上传文件接口"""
    # 设置单个文件大小限制为 500MB
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    
    result = []
    for file in files:
        try:
            task_id = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(file)}"
            upload_manager.add_task(task_id, file.filename, file.size)
            
            # 检查文件大小
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            
            if size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件 {file.filename} 超过大小限制 (500MB)"
                )
                
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = file.filename
            file_extension = os.path.splitext(original_filename)[1].lower()
            base_filename = os.path.splitext(original_filename)[0]

            # 处理压缩文件
            if file_extension in ['.zip', '.rar', '.7z']:
                # 创建临时目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 保存上传的压缩文件
                    temp_file = os.path.join(temp_dir, original_filename)
                    contents = await file.read()
                    with open(temp_file, 'wb') as f:
                        f.write(contents)

                    # 解压文件
                    extract_dir = os.path.join(temp_dir, 'extracted')
                    os.makedirs(extract_dir, exist_ok=True)

                    try:
                        # 解压文件
                        if file_extension == '.zip':
                            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                        elif file_extension == '.rar':
                            with rarfile.RarFile(temp_file, mode='r') as rar_ref:
                                rar_ref.extractall(extract_dir)
                        elif file_extension == '.7z':
                            with py7zr.SevenZipFile(temp_file, 'r') as sz_ref:
                                sz_ref.extractall(extract_dir)

                        # 检查解压后的结构
                        extracted_items = os.listdir(extract_dir)
                        
                        # 果只有一个文件夹，直接使用该文件夹作为根目录
                        if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_items[0])):
                            root_dir = os.path.join(extract_dir, extracted_items[0])
                            base_folder = extracted_items[0]  # 使用实际的文件夹名
                        else:
                            # 有多个文件/文件夹，创建以压缩包命名的根目录
                            root_dir = os.path.join(extract_dir, base_filename)
                            base_folder = base_filename  # 使用压缩包名作为文件夹名
                            os.makedirs(root_dir, exist_ok=True)
                            for item in extracted_items:
                                src_path = os.path.join(extract_dir, item)
                                dst_path = os.path.join(root_dir, item)
                                shutil.move(src_path, dst_path)

                        # 处理所有图片文件，保持目录结构
                        for root, _, files in os.walk(root_dir):
                            for img_file in files:
                                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                                    # 获取相对路径
                                    rel_path = os.path.relpath(root, root_dir)
                                    if rel_path == '.':
                                        target_dir = os.path.join(CUSTOM_UPLOAD_DIR, base_folder)
                                    else:
                                        target_dir = os.path.join(CUSTOM_UPLOAD_DIR, base_folder, rel_path)
                                    
                                    os.makedirs(target_dir, exist_ok=True)
                                    
                                    # 复制文件
                                    src_path = os.path.join(root, img_file)
                                    new_filename = f"{timestamp}_{img_file}"
                                    dst_path = os.path.join(target_dir, new_filename)
                                    
                                    shutil.copy2(src_path, dst_path)
                                    
                                    # 构建相对URL路径
                                    rel_url_path = os.path.relpath(dst_path, CUSTOM_UPLOAD_DIR)
                                    result.append({
                                        "image_id": new_filename,
                                        "original_name": img_file,
                                        "url": f"/static/uploads/{rel_url_path.replace(os.sep, '/')}",
                                        "folder": os.path.dirname(rel_url_path).replace(os.sep, '/')
                                    })

                    except Exception as e:
                        print(f"解压失败: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"解压文件失败: {str(e)}"
                        )
            else:
                # 处理个图片文件，直接
                filename = f"{timestamp}_{original_filename}"
                filepath = os.path.join(CUSTOM_UPLOAD_DIR, filename)
                
                contents = await file.read()
                with open(filepath, "wb") as f:
                    f.write(contents)
                
                result.append({
                    "image_id": filename,
                    "original_name": original_filename,
                    "url": f"/static/uploads/{filename}",
                    "folder": ""  # 单个文件没有文件夹
                })

            # 更新上传进度
            upload_manager.update_task(task_id, 100)
            upload_manager.complete_task(task_id, "success")
            
        except Exception as e:
            print(f"上传失败: {str(e)}")
            if 'task_id' in locals():
                upload_manager.complete_task(task_id, "error")
            raise HTTPException(status_code=500, detail=str(e))
            
    return {"message": "上传成功", "files": result}

@app.get("/api/images")
async def list_images(page: int = 1, page_size: int = 20):
    """获取图片表"""
    try:
        all_images = [f for f in os.listdir(CUSTOM_UPLOAD_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        total = len(all_images)
        
        # 分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        images = all_images[start_idx:end_idx]
        
        result = []
        for img in images:
            result.append({
                "image_id": img,
                "url": f"/static/uploads/{img}",
                "annotations": load_annotation(img)  # 加载已有的标注
            })
            
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "images": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片列表失败: {str(e)}")

@app.post("/api/save-annotation")
async def save_annotation(
    image_id: str = Form(...),
    labels: str = Form(...),
    prompt: str = Form(None),
    negative_prompt: str = Form(None)
):
    """保存标注数据"""
    try:
        annotation_data = {
            "labels": json.loads(labels) if isinstance(labels, str) else labels,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "updated_at": datetime.now().isoformat()
        }
        
        # 保存标注到JSON文件
        annotation_file = os.path.join(CUSTOM_ANNOTATION_DIR, f"{image_id}.json")
        with open(annotation_file, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
        return {"message": "标注保存成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存标注失败: {str(e)}")

def load_annotation(image_id: str) -> dict:
    """加载图片的标注数据"""
    annotation_file = os.path.join(CUSTOM_ANNOTATION_DIR, f"{image_id}.json")
    if os.path.exists(annotation_file):
        with open(annotation_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# 添加 WebSocket 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.training_tasks: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def broadcast_training_progress(self, task_id: str, progress: float, status: str):
        """广播训练进度"""
        message = {
            "type": "training_progress",
            "task_id": task_id,
            "progress": progress,
            "status": status
        }
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except Exception:
                continue

    async def send_personal_message(self, client_id: str, message: dict):
        """发送个人消息"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()

# WebSocket 路由
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理其他 WebSocket 信息...
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        manager.disconnect(client_id)

# 添加训练队列管理器
class TrainingQueueManager:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.current_task = None
        self.running = False
    
    async def add_task(self, task_info):
        """添加训练任务到队列"""
        await self.queue.put(task_info)
        if not self.running:
            asyncio.create_task(self.process_queue())
    
    async def process_queue(self):
        """处理训练队列"""
        self.running = True
        while True:
            try:
                if self.queue.empty():
                    self.running = False
                    break
                
                # 获取下一个任务
                task_info = await self.queue.get()
                self.current_task = task_info
                
                # 启动训练进程
                process = await self.start_training_process(task_info)
                
                # 等待训练完成
                while True:
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(1)
                
                # 更新任务状态
                task_info["status"] = "completed"
                self.current_task = None
                
                # 标记任务完成
                self.queue.task_done()
                
            except Exception as e:
                print(f"Training error: {str(e)}")
                if self.current_task:
                    self.current_task["status"] = "failed"
                    self.current_task["error"] = str(e)
    
    async def start_training_process(self, task_info):
        """启动训练进程"""
        sd_scripts_dir = config["custom_sd_scripts_dir"]
        original_dir = os.getcwd()
        os.chdir(sd_scripts_dir)
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env["PYTHONPATH"] = sd_scripts_dir
            env["CUDA_LAUNCH_BLOCKING"] = "1"  # 强制同步 CUDA 操作
            
            process = subprocess.Popen([
                sys.executable,
                "sdxl_train_network.py",
                "--config_file", task_info["config_file"],
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
               env=env, bufsize=1, universal_newlines=True)
            
            # 启动错误输出监控
            asyncio.create_task(self.monitor_error_output(process, task_info))
            # 启动标准输出监控
            asyncio.create_task(self.monitor_process_output(process, task_info))
            
            return process
            
        finally:
            os.chdir(original_dir)
    
    async def monitor_error_output(self, process, task_info):
        """监控错误输出"""
        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None, process.stderr.readline
            )
            if not line:
                break
            line = line.strip()
            if line:
                print(f"Training error output [{task_info['task_id']}]: {line}")
                # 如果发现严重错误，可以更新任务状态
                if "error" in line.lower() or "exception" in line.lower():
                    task_info["status"] = "error"
                    task_info["error"] = line
    
    async def monitor_process_output(self, process, task_info):
        """监控进程输出"""
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, process.stdout.readline
                )
                if not line:
                    break
                line = line.strip()
                if line:
                    print(f"Training output [{task_info['task_id']}]: {line}")
                    
                    # 更新进度信息
                    if "epoch" in line.lower():
                        # 解析进度信息并更新
                        task_info["progress"] = self.parse_progress(line)
                        # 广播进度
                        await manager.broadcast_training_progress(
                            task_info["task_id"],
                            task_info["progress"],
                            task_info["status"]
                        )
            except Exception as e:
                print(f"Error monitoring output: {str(e)}")
                break
    
    def parse_progress(self, line):
        """解析训练输出中的进度信息"""
        try:
            # 根据实际输出格式修改解析逻辑
            if "epoch" in line.lower():
                # 示例: 假设输出格式为 "Epoch 1/100"
                current, total = re.findall(r"epoch (\d+)/(\d+)", line.lower())[0]
                return (int(current) / int(total)) * 100
        except:
            pass
        return 0

# 创建训练队列管理器实例
training_queue = TrainingQueueManager()

# 修改启动训练的路由
@app.post("/api/start-training")
async def start_training(request: Request):
    """启动模型训练"""
    try:
        data = await request.json()
        settings = data.get("settings", {})
        folder = data.get("folder")
        
        print(f"Starting training for folder: {folder}")
        print(f"Settings: {settings}")
        
        # 检查临时配置文件目录
        sd_scripts_dir = config["custom_sd_scripts_dir"]
        temp_config_dir = os.path.join(sd_scripts_dir, "temp_configs")
        if not os.path.exists(temp_config_dir):
            raise HTTPException(status_code=404, detail="未找到训练配置文件夹")
        
        # 查找对应文件夹的配置文件
        config_files = []
        for file in os.listdir(temp_config_dir):
            if file.startswith(f"train_{folder}_") and file.endswith(".toml"):
                config_files.append(file)  # 只保存文件名
        
        if not config_files:
            raise HTTPException(status_code=404, detail="未找到训练配置文件，请先保存训练设置")
        
        # 获取第一个配置文件
        config_name = config_files[0]
        config_file = os.path.join(temp_config_dir, config_name)  # 完整路径
        print(f"Starting training with config: {config_file}")
        
        # 创建任务信息
        task_id = f"task_{int(time.time())}"
        task_info = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "model_name": settings.get("baseModel"),
            "folder": folder,
            "config_file": os.path.join("temp_configs", config_name),
            "start_time": datetime.now().isoformat(),
            "output_dir": os.path.join(config["custom_sd_scripts_dir"], "output"),  # 添加输出目录
            "params": {
                "learning_rate": settings.get("learningRate"),
                "steps": settings.get("steps"),
                "batch_size": settings.get("batchSize"),
                "network_dim": settings.get("networkDim", 8)
            }
        }
        
        # 添加到训练队列
        await training_queue.add_task(task_info)
        
        # 保存到任务列表
        manager.training_tasks[task_id] = task_info
        
        return {
            "message": "训练任务已加入队列",
            "task_id": task_id,
            "position": training_queue.queue.qsize(),
            "status": "queued"
        }
        
    except Exception as e:
        print(f"Start training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")

@app.get("/api/training-status/{task_id}")
async def get_training_status(task_id: str):
    """获取训练状态"""
    # TODO: 获取训练进度
    return {"status": "training", "progress": 0.5}

@app.get("/")
async def root():
    return FileResponse(os.path.join(CUSTOM_STATIC_DIR, "index.html"))

# 添加获取文件结构的路由
@app.get("/api/files")
async def list_files(path: str = ""):
    """获取文件结构"""
    try:
        base_path = Path(CUSTOM_STATIC_DIR)
        current_path = base_path / path if path else base_path
        
        # 确保路径在允许的范围内
        if not str(current_path.resolve()).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="访问被拒绝")
            
        if not current_path.exists():
            raise HTTPException(status_code=404, detail="路径不存在")
            
        items = []
        for item in current_path.iterdir():
            is_dir = item.is_dir()
            items.append({
                "name": item.name,
                "path": str(item.relative_to(base_path)),
                "type": "directory" if is_dir else "file",
                "size": "" if is_dir else get_file_size(item),
                "modified": get_file_modified_time(item)
            })
            
        return {
            "current_path": path,
            "items": sorted(items, key=lambda x: (x["type"] == "file", x["name"]))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

def get_file_size(file_path: Path) -> str:
    """获取文件大小的人类可读格式"""
    size_bytes = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def get_file_modified_time(file_path: Path) -> str:
    """获取文件修改时间"""
    timestamp = file_path.stat().st_mtime
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

# 添加安全检查装饰器
def validate_path(path: str) -> bool:
    """验���路径是否安全"""
    # 规范化路径
    normalized_path = os.path.normpath(path)
    # 检查是否试图访问上级目录
    if '..' in normalized_path:
        return False
    # 检查是否是绝对路径
    if os.path.isabs(normalized_path):
        return False
    return True

def secure_path(func):
    async def wrapper(*args, **kwargs):
        paths = kwargs.get('paths', [])
        if not all(validate_path(path) for path in paths):
            raise HTTPException(
                status_code=400,
                detail="检测到不安全的文件路径"
            )
        return await func(*args, **kwargs)
    return wrapper

# 使用装饰器
@app.post("/api/upload-folder")
@secure_path
async def upload_folder(
    files: List[UploadFile] = File(...),
    paths: List[str] = Form(...)
):
    """上传文件夹"""
    try:
        result = []
        for file, path in zip(files, paths):
            # 构建目标路径
            target_path = os.path.join(CUSTOM_STATIC_DIR, os.path.dirname(path))
            
            # 确保目标目录存在
            os.makedirs(target_path, exist_ok=True)
            
            # 构建完整的文件路径
            file_path = os.path.join(target_path, os.path.basename(path))
            
            # 保存文件
            try:
                contents = await file.read()
                with open(file_path, "wb") as f:
                    f.write(contents)
                
                result.append({
                    "path": path,
                    "size": len(contents)
                })
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"保存文件 {path} 失败: {str(e)}"
                )
            finally:
                await file.close()
                
        return {
            "message": "文件夹上传成功",
            "files": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件夹上传失败: {str(e)}"
        )

@app.get("/api/preview-folder")
async def preview_folder(folder_path: str = ""):
    """���览文件夹内的图片"""
    try:
        base_path = Path(CUSTOM_STATIC_DIR)
        preview_path = base_path / folder_path if folder_path else base_path
        
        # 确保路径在允许的范围内
        if not str(preview_path.resolve()).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="访问被拒绝")
            
        if not preview_path.exists():
            raise HTTPException(status_code=404, detail="路径不存在")
            
        # 递获取所有图片文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.gif']:
            image_files.extend(preview_path.rglob(f"*{ext}"))
            image_files.extend(preview_path.rglob(f"*{ext.upper()}"))
        
        # 构建图片信息
        images = []
        for img_path in image_files:
            relative_path = img_path.relative_to(base_path)
            images.append({
                "name": img_path.name,
                "path": str(relative_path),
                "url": f"/static/{relative_path}",
                "size": get_file_size(img_path),
                "modified": get_file_modified_time(img_path)
            })
            
        return {
            "folder_path": folder_path,
            "total": len(images),
            "images": sorted(images, key=lambda x: x["name"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片预览失败: {str(e)}")

@app.get("/workspace/{folder_path:path}")
async def workspace(folder_path: str):
    """回工作空间页面"""
    return FileResponse(os.path.join(CUSTOM_STATIC_DIR, "workspace.html"))

@app.get("/api/workspace/images/{path:path}")
async def get_workspace_images(request: Request, path: str = ""):
    try:
        # 获请求来源
        referer = request.headers.get('referer', '')
        print(f"Referer: {referer}")
        
        # 获取文件和文件夹列表
        items = []
        
        # 如果是从 index.html  / 访问
        if '/static/index.html' in referer or referer.endswith('/'):
            # 返回所有顶层文件夹
            for item in Path(CUSTOM_UPLOAD_DIR).iterdir():
                if item.is_dir():
                    rel_path = str(item.relative_to(CUSTOM_UPLOAD_DIR))
                    print(f"Found root folder: {rel_path}")
                    items.append({
                        "name": item.name,
                        "path": rel_path,
                        "type": "directory",
                        "level": 0,
                        "parent": None,
                        "has_children": any(subitem.is_dir() for subitem in item.iterdir())
                    })
        # 如果是从工作训练页面访问
        elif '/workspace/' in referer or '/training/' in referer:
            # 从 referer ���取当前工作区文件夹名
            workspace_match = re.search(r'/(?:workspace|training)/([^/]+)', referer)
            if workspace_match:
                current_workspace = workspace_match.group(1)
                print(f"Current workspace: {current_workspace}")
                
                # 获取当前工作区的完整路径
                workspace_path = Path(CUSTOM_UPLOAD_DIR) / current_workspace
                if workspace_path.exists() and workspace_path.is_dir():
                    # 添加根文件夹
                    items.append({
                        "name": current_workspace,
                        "path": current_workspace,
                        "type": "directory",
                        "level": 0,
                        "parent": None,
                        "has_children": any(item.is_dir() for item in workspace_path.iterdir())
                    })
                    
                    # 获取所有子文件夹
                    for subfolder in workspace_path.iterdir():
                        if subfolder.is_dir():
                            rel_path = str(subfolder.relative_to(CUSTOM_UPLOAD_DIR))
                            items.append({
                                "name": subfolder.name,
                                "path": rel_path,
                                "type": "directory",
                                "level": 1,
                                "parent": current_workspace,
                                "has_children": any(subitem.is_dir() for subitem in subfolder.iterdir())
                            })
                            
                            # 获取所有子子文件夹
                            for subsubfolder in subfolder.iterdir():
                                if subsubfolder.is_dir():
                                    rel_path = str(subsubfolder.relative_to(CUSTOM_UPLOAD_DIR))
                                    items.append({
                                        "name": subsubfolder.name,
                                        "path": rel_path,
                                        "type": "directory",
                                        "level": 2,
                                        "parent": str(subfolder.relative_to(CUSTOM_UPLOAD_DIR)),
                                        "has_children": any(item.is_dir() for item in subsubfolder.iterdir())
                                    })
        
        print(f"Found items: {items}")
        return {"images": items}
        
    except Exception as e:
        print(f"Error in get_workspace_images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@app.post("/api/workspace/batch-annotate")
async def batch_annotate(
    image_paths: List[str] = Body(...),
    prompt: str = Body(None),
    negative_prompt: str = Body(None),
    tags: List[str] = Body(...)
):
    """批量标注图"""
    try:
        for path in image_paths:
            annotation_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "tags": tags,
                "updated_at": datetime.now().isoformat()
            }
            
            image_id = os.path.basename(path)
            annotation_file = os.path.join(CUSTOM_ANNOTATION_DIR, f"{image_id}.json")
            
            with open(annotation_file, "w", encoding="utf-8") as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
                
        return {"message": "批量标注成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量标注失败: {str(e)}")

# 添加上传任务管理
class UploadTaskManager:
    def __init__(self):
        self.active_tasks = {}  # 当前活动的任务
        self.history = deque(maxlen=100)  # 最近100条历史记录
        self.connected_clients = set()  # WebSocket 客户端
        
    def add_task(self, task_id: str, filename: str, size: int, user_id: str = "anonymous"):
        """添加新的上传任务"""
        task = {
            "id": task_id,
            "filename": filename,
            "size": size,
            "progress": 0,
            "status": "uploading",
            "user_id": user_id,
            "start_time": datetime.now().isoformat(),
            "speed": 0,
            "remaining": None
        }
        self.active_tasks[task_id] = task
        asyncio.create_task(self.broadcast_tasks())
        
    def update_task(self, task_id: str, progress: int, speed: float = 0):
        """更新任务进度"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task["progress"] = progress
            task["speed"] = speed
            if speed > 0:
                remaining_bytes = task["size"] * (100 - progress) / 100
                task["remaining"] = remaining_bytes / speed
            asyncio.create_task(self.broadcast_tasks())
            
    def complete_task(self, task_id: str, status: str = "success"):
        """完成任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task["status"] = status
            task["end_time"] = datetime.now().isoformat()
            task["progress"] = 100 if status == "success" else task["progress"]
            self.history.appendleft(task)
            del self.active_tasks[task_id]
            asyncio.create_task(self.broadcast_tasks())
            
    def get_all_tasks(self):
        """获取所有任务状态"""
        return {
            "active": list(self.active_tasks.values()),
            "history": list(self.history)
        }
        
    async def broadcast_tasks(self):
        """广播任务状态到所有连接的客户端"""
        message = {
            "type": "upload_status",
            "data": self.get_all_tasks()
        }
        for client in self.connected_clients:
            try:
                await client.send_json(message)
            except Exception:
                pass

# 创建全局任务管理器实例
upload_manager = UploadTaskManager()

# 添加自动打标相关的路由
@app.post("/api/auto-tag")
async def auto_tag(request: Request):
    """自动打标接口"""
    try:
        data = await request.json()
        folder = data["folder"]
        model = data["model"]
        folder_settings = data.get("folder_settings", {})

        async def generate_progress():
            try:
                folder_path = Path(CUSTOM_UPLOAD_DIR) / folder
                if not folder_path.exists():
                    yield json.dumps({"error": "文件夹不存在"}) + "\n"
                    return

                # 获取所有子文件夹
                subfolders = [folder_path] + [f for f in folder_path.rglob("*") if f.is_dir()]
                total_folders = len(subfolders)
                total_processed = 0
                results = []

                # 按路径长度排序，确保文件夹先处理
                subfolders.sort(key=lambda x: len(str(x)))

                # 初始化累积设置字典
                accumulated_settings = {}
                
                # 第一步：计算所有文件夹的继承设置
                yield json.dumps({
                    "progress": 0,
                    "info": "正在计算文件夹设置继承关系..."
                }) + "\n"

                # 先添加根文件夹的设置
                root_path = folder
                if root_path in folder_settings:
                    accumulated_settings[root_path] = folder_settings[root_path]
                    print(f"Root settings: {accumulated_settings[root_path]}")

                # 计算所有文件夹的继承设置
                for subfolder in subfolders:
                    rel_path = str(subfolder.relative_to(Path(CUSTOM_UPLOAD_DIR)))
                    parent_path = str(subfolder.parent.relative_to(Path(CUSTOM_UPLOAD_DIR)))
                    
                    # 获取当前文件夹的设置
                    current_folder_setting = folder_settings.get(rel_path, {})
                    
                    # 合并父文件夹的设置
                    merged_settings = {}
                    if parent_path in accumulated_settings:
                        parent_settings = accumulated_settings[parent_path]
                        print(f"Merging settings for {rel_path} with parent {parent_path}")
                        print(f"Parent settings: {parent_settings}")
                        print(f"Current settings: {current_folder_setting}")
                        
                        for tag_type in ['character', 'copyright', 'artist', 'special']:
                            parent_tags = parent_settings.get(tag_type, "")
                            current_tags = current_folder_setting.get(tag_type, "")
                            merged_settings[tag_type] = f"{parent_tags}, {current_tags}" if parent_tags and current_tags else parent_tags or current_tags
                    else:
                        merged_settings = current_folder_setting

                    # 保存当前文件夹的累积设置
                    accumulated_settings[rel_path] = merged_settings
                    print(f"Final settings for {rel_path}: {merged_settings}")

                # 第二步：用计算好的设置进行打标
                yield json.dumps({
                    "progress": 10,
                    "info": f"开始处理 {total_folders} 个文件夹..."
                }) + "\n"

                # 理每个文件夹
                for idx, subfolder in enumerate(subfolders, 1):
                    progress = 10 + (idx / total_folders * 90)  # 从10%到100%
                    yield json.dumps({
                        "progress": progress,
                        "info": f"正在处理第 {idx}/{total_folders} 个文件夹: {subfolder.name}"
                    }) + "\n"

                    # 检查文件夹是否包含图片
                    has_images = any(f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp'] 
                                   for f in subfolder.iterdir() if f.is_file())
                    
                    if has_images:
                        try:
                            rel_path = str(subfolder.relative_to(Path(CUSTOM_UPLOAD_DIR)))
                            merged_settings = accumulated_settings.get(rel_path, {})
                            
                            # 调用标签识别函数
                            result = traverse_tag_images(
                                folder_path=str(subfolder),
                                model_name=model,
                                add_character=merged_settings.get("character", ""),
                                add_copyright=merged_settings.get("copyright", ""),
                                add_artist=merged_settings.get("artist", ""),
                                add_special=merged_settings.get("special", "")
                            )
                            
                            if result:
                                total_processed += result.get("processed_count", 0)
                                results.append({
                                    "folder": rel_path,
                                    "processed": result.get("processed_count", 0),
                                    "settings": merged_settings,
                                    "details": result
                                })

                        except Exception as e:
                            yield json.dumps({
                                "error": f"处理文件夹 {subfolder} 失败: {str(e)}"
                            }) + "\n"

                # 发送最终结果
                yield json.dumps({
                    "progress": 100,
                    "info": "处理完成",
                    "total_processed": total_processed,
                    "folder_results": results
                }) + "\n"

            except Exception as e:
                yield json.dumps({
                    "error": f"处理过程中发生错误: {str(e)}"
                }) + "\n"

        return StreamingResponse(
            generate_progress(),
            media_type="application/x-ndjson"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"自动打标失败: {str(e)}")

@app.post("/api/save-training-settings")
async def save_training_settings(request: Request):
    """保存训练设置"""
    try:
        data = await request.json()
        folder = data.get("folder")
        settings = data.get("settings", {})
        
        print(f"Received settings request for folder: {folder}")
        print(f"Settings: {settings}")
        
        if not folder:
            raise HTTPException(status_code=400, detail="未提供文件夹名称")
        
        # 获取训练参数
        learning_rate = settings.get("learningRate", 0.0001)
        max_train_epochs = settings.get("steps", 20)
        network_dim = settings.get("networkDim", 8)
        
        # 构建完整的数据目录路径
        data_dir = os.path.join(CUSTOM_UPLOAD_DIR, folder)
        if not os.path.exists(data_dir):
            raise HTTPException(status_code=404, detail=f"文件夹 {folder} 不存在")
        
        # 创建临时配置文件目录（在 sd-scripts 目录下）
        sd_scripts_dir = config["custom_sd_scripts_dir"]
        temp_config_dir = os.path.join(sd_scripts_dir, "temp_configs")
        os.makedirs(temp_config_dir, exist_ok=True)
        
        # 读取基础 toml 文件（从 sd-scripts 目录）
        base_toml_path = os.path.join(sd_scripts_dir, "train_neta_config.toml")
        print(f"Reading base toml file: {base_toml_path}")
        
        if not os.path.exists(base_toml_path):
            raise HTTPException(status_code=404, detail=f"基础配置文件不存在: {base_toml_path}")
        
        # 为每个子文件夹创建配置文件
        config_files = []
        for subdir in [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]:
            # 生成配置文件名
            config_name = f"train_{folder}_{subdir}.toml"
            config_path = os.path.join(temp_config_dir, config_name)
            
            # 先复制基础配置文件
            shutil.copy2(base_toml_path, config_path)
            print(f"Copied base config to: {config_path}")
            
            # 修改配置文件
            edit_toml(
                data_dir=os.path.join(data_dir, subdir),  # 完整的子文件夹路径
                toml_file=config_path,  # 新的配置文件路径
                pretrained_model_name_or_path=settings.get("baseModel", ""),
                end=settings.get("modelSuffix", "_lora"),
                learning_rate=learning_rate,
                max_train_epochs=max_train_epochs,
                network_dim=network_dim
            )
            config_files.append(config_path)
            print(f"Created config file: {config_path} for subfolder: {subdir}")
        
        return {
            "message": "训练设置保存",
            "config_files": config_files
        }
    except Exception as e:
        print(f"Save training settings error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存训练设置失败: {str(e)}")

# 添加一个新的路由来获取基础模型列表
@app.get("/api/base-models")
async def get_base_models():
    """获取基础模型列表"""
    try:
        # 检查配置是否存在
        if "custom_pretrained_model_dir" not in config:
            print("Missing custom_pretrained_model_dir in config")
            print("Available config keys:", config.keys())
            raise HTTPException(
                status_code=500, 
                detail="预训练模型目录未配置"
            )
        
        model_dir = Path(config["custom_pretrained_model_dir"])
        print(f"Looking for models in: {model_dir}")
        
        if not model_dir.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"预训练模型目录不存在: {model_dir}"
            )
        
        # 获取所有 safetensors  ckpt 文件
        model_files = []
        for ext in ['.safetensors', '.ckpt']:
            found_files = list(model_dir.glob(f'*{ext}'))
            print(f"Found {len(found_files)} {ext} files")
            model_files.extend(found_files)
        
        # 构建模型列表
        models = []
        for model_file in model_files:
            model_info = {
                "label": model_file.stem,
                "value": str(model_file),
                "type": model_file.suffix[1:]
            }
            print(f"Adding model: {model_info}")
            models.append(model_info)
        
        return {"models": models}
    except Exception as e:
        print(f"Get base models error: {str(e)}")
        print(f"Config: {config}")
        raise HTTPException(
            status_code=500, 
            detail=f"获取基础模型列表失败: {str(e)}"
        )

@app.get("/api/training-tasks")
async def get_training_tasks():
    """获取所有训练任务"""
    try:
        tasks = []
        for task_id, task in manager.training_tasks.items():
            tasks.append({
                "task_id": task_id,
                "status": task.get("status", "unknown"),
                "progress": task.get("progress", 0),
                "model_name": task.get("model_name", ""),
                "folder": task.get("folder", ""),
                "start_time": task.get("start_time", ""),
                "error": task.get("error", ""),
                "logs": task.get("logs", [])
            })
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练任务失败: {str(e)}")

@app.post("/api/stop-training/{task_id}")
async def stop_training(task_id: str):
    """停止训练任务"""
    try:
        if task_id in manager.training_tasks:
            task = manager.training_tasks[task_id]
            if task.get("process"):
                task["process"].terminate()
                task["status"] = "stopped"
                return {"message": "训练已停止"}
        raise HTTPException(status_code=404, detail="未找到训练任务")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止训练失败: {str(e)}")

@app.post("/api/open-dir")
async def open_directory(path: str):
    """打开目录"""
    try:
        # 检查路径是否存在
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="目录不存在")
            
        # 根据操作系统打开目录
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
            
        return {"message": "目录已打开"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打开目录失败: {str(e)}")

@app.get("/api/list-files")
async def list_files(path: str):
    """列出目录中的文件"""
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="目录不存在")
            
        files = []
        for entry in os.scandir(path):
            file_info = {
                "name": entry.name,
                "path": str(entry.path),
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else 0,
                "modified": entry.stat().st_mtime
            }
            files.append(file_info)
            
        # 按类型和名称排序
        files.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
        
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@app.get("/api/download-file")
async def download_file(path: str):
    """下载文件"""
    try:
        if not os.path.exists(path) or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="文件不存在")
            
        return FileResponse(
            path,
            filename=os.path.basename(path),
            media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")

if __name__ == "__main__":
    # 使用配置文件的服务器设置
    uvicorn.run(
        app, 
        host=config["server"]["host"], 
        port=config["server"]["port"]
    )



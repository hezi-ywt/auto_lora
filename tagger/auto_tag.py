import json
import os
import re
# from waifuc.export import SaveExporter
# from waifuc.source import DanbooruSource
# import os
import random
from imgutils.tagging import wd14
from imgutils.tagging import drop_overlap_tags
from PIL import Image
# 读取 JSON 文件



def format_tag(tag):
    if len(tag) >= 4:
        tag = tag.replace("_", " ")
    return tag

def format_tags(tags):
    if isinstance(tags,str):
        tags = tags.split(", ")
    tags = [format_tag(tag) for tag in tags]
    tags = list(set(tags))
    tags = ", ".join(tags)
    return tags

def format_tags_str(tags_str):
    if isinstance(tags_str,str):
        tags_str = tags_str.split(",")
    for i in range(len(tags_str)):
        tags_str[i] = tags_str[i].strip()
    tags_str = ", ".join(tags_str)
    return tags_str

def get_meta_tag(data_file):

    with open(data_file, 'r') as file:
        json_data = file.read()

    # 解析 JSON 数据
    data = json.loads(json_data)

    # 打印解析后的数据
    return data["danbooru"]


def is_image(pth):
    img_endwith = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
    if pth.endswith(tuple(img_endwith)):
        return True
    else:
        return False

special_list=[
    "1girl",
    "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "multiple_girls",
    "1boy",
    "2boys",
    "3boys",
    "4boys",
    "5boys",
    "6+boys",
    "multiple_boys",
    "male_focus",
    "1other",
    "2others",
    "3others",
    "4others",
    "5others",
    "6+others",
    "multiple_others",
    "giantess",
    "miniboy",
    "minigirl",
    "giant",
     
]


# meta_keywords_black_list_path = "meta_tag_black_list.txt"
# with open(meta_keywords_black_list_path, "r") as f:
#     meta_keywords_black_list = f.read().splitlines()



def traverse_tag_images(folder_path, data_folder_path=None, model_name='ConvNextV2', add_special="", add_character="", add_artist="", add_copyright=""):
    """遍历文件夹中的图片并进行标注"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', ".webp", ".bmp"]
    processed_count = 0
    processed_files = []

    if data_folder_path is None:
        data_folder_path = folder_path
    folder_path_end = folder_path.split("/")[-1]

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name, extension = os.path.splitext(file)
            extension = extension.lower()

            # 判断文件是否是图片文件
            if extension in image_extensions:
                image_file = os.path.join(root, file)
                txt_file = os.path.join(root, file_name + '.txt')
                data_file = os.path.join(root, "." + file_name + '_meta.json')
                try:
                    if not os.path.exists(data_file):
                        # 根据模型名称选择对应的模型
                        if model_name == 'wd-swinv2-tagger-v3':
                            rating, feature, chars = wd14.get_wd14_tags(image_file, model_name='SwinV2', general_threshold=0.35, character_threshold=0.85)
                        elif model_name == 'wd-eva02-large-tagger-v3':
                            rating, feature, chars = wd14.get_wd14_tags(image_file, model_name='EVA02', general_threshold=0.35, character_threshold=0.85)
                        else:  # ConvNextV2
                            rating, feature, chars = wd14.get_wd14_tags(image_file, model_name='ConvNextV2', general_threshold=0.35, character_threshold=0.85)

                        # 创建 txt 文件
                        tags =  list(feature.keys())
                        sp = []
                        for i in special_list:
                            if i in tags:
                                sp.append(i)
                                tags.remove(i)
                                

                        chars = list(chars.keys())
                        # rating, a dict containing ratings and their confidences
                        #get max confidence rating
                        rating = max(rating, key=rating.get)
                        rate = rating
                        copyright = []
                        artist = []

                    else:
                        sp = []
                        data = get_meta_tag(data_file)
                        chars = data.get("tag_string_character","").split(" ")
                        copyright = data["tag_string_copyright"].split(" ")
                        artist = data["tag_string_artist"].split(" ")
                        tags = data["tag_string_general"].split(" ")
                        rate = ""
                        for i in special_list:
                            if i in tags:
                                sp.append(i)
                                tags.remove(i)

                            
                    
                    if add_character != "":
                        chars = format_tags_str(add_character).split(", ") + chars 
                    else:
                        chars = chars
                    if add_copyright != "":
                        copyright = format_tags_str(add_copyright).split(", ") + copyright 
                    else:
                        copyright = copyright
                        
                    if add_artist != "":
                        artist = format_tags_str(add_artist).split(", ") + artist
                    else:
                        artist = artist
                    
                    
                    if add_special != "":
                        tags = format_tags_str(add_special).split(", ") + tags
                    else:
                        tags = tags

                    for n in range(len(chars)):
                        if len(chars[n]) >= 4:
                            chars[n] = chars[n].replace("_", " ")
                    for n in range(len(artist)):
                        if len(artist[n]) >= 4:
                            artist[n] = artist[n].replace("_", " ")
                    for n in range(len(sp)):
                        if len(sp[n]) >= 4:
                            sp[n] = sp[n].replace("_", " ")
                    for n in range(len(tags)):
                        if len(tags[n]) >= 4:
                            tags[n] = tags[n].replace("_", " ")
                    
                    chars = list(set(chars))
                    artist = list(set(artist))
                    copyright = list(set(copyright))
                    tags = list(set(tags))
                    
                    
                                
                    tags = ", ".join(tag for tag in tags)
                    chars = ", ".join(tag for tag in chars)
                    artist = ", ".join(tag  for tag in artist)
                    copyright = ", ".join(tag for tag in copyright)
                    sp = ", ".join(tag for tag in sp)
                    
                    for i in [chars,copyright,artist]:
                        if i == "":
                            continue
                        else:
                            sp = sp + ", " + i
                    
                        
                    tags =  sp + ", |||" + tags + ", " + rate

                    




                    # tags = tags.replace("_", " ")
                        
                        
                    # if  not folder_path_end.endswith("reg"):
                        
                    #     tags = folder_path_end.split("_",1)[1] + ", " + tags
                                    
                                
                    with open(txt_file, 'w') as file:
                        file.write(tags)
                    
                    processed_count += 1
                    processed_files.append({
                        "file": image_file,
                        "tags": tags
                    })
                    
                    print(f"Created {txt_file}")
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    # 返回处理结果
    return {
        "processed_count": processed_count,
        "processed_files": processed_files,
        "folder": folder_path
    }


# def down_batch(names,is_style=False):
#     for name in names:
#         name1 = name.replace("(","")
#         name1 = name1.replace(")","")
#         if is_style:
#             out_dir = "train/"+name1+"_style/5_"+name1+"_artstyle"
#         else:
#             out_dir =  "train/"+name1+"/5_"+name1

#         os.makedirs(out_dir, exist_ok=True)
#         source = DanbooruSource([name])
#         source.attach(
#             # # only 1 head,
#             #HeadCountAction(1),
#             # # if shorter side is over 640, just resize it to 640
#             # AlignMinSizeAction(640),
#         ).export(  # only first 10 images
#             # save images (with meta information from danbooru site)
#             SaveExporter(out_dir)
#         )

#         traverse_tag_images(out_dir)
        
def rotate_img(img, angle=180):
    # 旋转图片
    img = img.rotate(angle)
    return img        

def flip_img(img):
    # 翻转图片
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
def rotate_aug(img_file):
    img = Image.open(img_file)
    img = rotate_img(img)

    img = rotate_img(img)
    img.save(os.path.join(os.path.dirname(img_file),  os.path.basename(img_file).split(".")[0]+"_rotated.jpg"))
    # txt_pth = os.join(os.path.dirname(img_file),  os.path.basename(img_file).split(".")[0]+".txt")
    # if os.path.exists(txt_pth):
    #     with open(txt_pth,"r") as f:
    #         txt = f.read()
    #     txt += ", upside-down" 
    #     with open(txt_pth,"w") as f:
    #         f.write(txt)

import random

def ran_aug(dir,ran=8):
    for root, dirs, files in os.walk(dir):
        img_file_list=[]
        for file in files:
            file_name, extension = os.path.splitext(file)
            extension = extension.lower()  # 将后缀转换为小写
            if extension in [".jpg",".jpeg",".png",".webp",".bmp"]:
                img_file = os.path.join(root, file)
                img_file_list.append(img_file)
            
            ran_choose = random.sample(img_file_list,ran)
            

from pathlib import Path
import re

def clean_filename(original_name):
    # 用下划线替换不合法字符，包括特殊字符和中文符号
    cleaned_name = re.sub(r'[<>:"/\\|?*]', '_', original_name)  # 替换不合法字符
    cleaned_name = re.sub(r'_{2,}', '_', cleaned_name)  # 替换多个下划线为一个
    return cleaned_name

def rename_files_in_directory(dir_path):
    # 创建 Path 对象
    path = Path(dir_path)

    # 检查目录是否存在
    if not path.is_dir():
        print("指定的路径不是一有效的目录。")
        return

    # 遍历目录下的所有文件
    for count, file in enumerate(path.glob("*"), start=1):
        if file.is_file():  # 确保是文件
            # 清理文件名
            cleaned_name = clean_filename(file.name)
            cleaned_name = f"{count}_{cleaned_name}"
            # 构造新文件名
            new_file_name = f"new_{cleaned_name}"
            new_file = path / new_file_name

            # 重命名文件
            try:
                file.rename(new_file)
                print(f"已重命名: {file} -> {new_file}")
            except PermissionError:
                print(f"权限错误: 无法重命名 {file}.")
            except Exception as e:
                print(f"错误: {e}")


if __name__ == '__main__':
    dir = "/mnt/e/data/train/napoli_azur_lane"
    for subdir in os.listdir(dir):
        if os.path.isdir(os.path.join(dir,subdir)):
            dir1 = os.path.join(dir,subdir)

            add_character = subdir.split("_",maxsplit=1)[-1]

            traverse_tag_images(dir1, add_character=add_character,add_copyright="azur_lane",)
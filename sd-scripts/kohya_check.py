import os
import json

img_exts = ['.jpg', '.jpeg', '.png', '.webp']
def is_image_file(file_name):
    return any(file_name.lower().endswith(ext) for ext in img_exts)

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"removed file: {file_path}")
    else:
        print(f"file not found: {file_path}")

def check_dir(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file):
                basename = os.path.basename(file)
                txt_name = basename.split('.')[0] + '.txt'
                txt_path = os.path.join(root, txt_name)
                if not os.path.exists(txt_path):
                    print(f"missing txt file: {txt_path}")
                else:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        txt = f.read()
                        if not txt:
                            print(f"empty txt file: {txt_path}")
                            remove_file(txt_path)
                            remove_file(file_path)
                            continue
                        if txt == '':
                            print(f"None txt file: {txt_path}")
                            remove_file(txt_path)
                            remove_file(file_path)
                            continue

def check_same_name(dir_path):
    name = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file):
                basename = os.path.basename(file).split('.')[0]
                if basename in name:
                    print(f"same name: {basename}")
                    
                else:
                    name.append(basename)
if __name__ == "__main__":
    check_dir('/mnt/e/data/train/fancy_azur_lane')
    
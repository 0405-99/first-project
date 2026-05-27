# preprocess.py
import os
import random
from tqdm import tqdm
import regex as re
from config import Config

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\u200b\ufeff]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_thucnews():
    cfg = Config()
    
    if not os.path.exists(cfg.raw_data_path):
        print(f"Error: 未找到原始数据文件夹 {cfg.raw_data_path}")
        return

    # =========================================================
    # 【核心修改】不再手动列出，而是自动扫描文件夹
    # =========================================================
    print("正在扫描类别文件夹...")
    # 获取目录下所有的文件夹名字，过滤掉隐藏文件
    target_classes = [
        d for d in os.listdir(cfg.raw_data_path) 
        if os.path.isdir(os.path.join(cfg.raw_data_path, d))
    ]
    # 排序一下，保证每次顺序一致
    target_classes.sort()
    
    print(f"✅ 成功识别到 {len(target_classes)} 个类别: {target_classes}")
    # =========================================================
    
    data_list = []
    
    print("开始清洗并加载原始数据...")
    for label in tqdm(target_classes):
        folder_path = os.path.join(cfg.raw_data_path, label)
        
        files = os.listdir(folder_path)
        # 【注意】每类只取 500 条用于快速训练
        # 如果你想跑全量，把下面这行注释掉即可
        files = files[:500] 
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content = clean_text(content)
                    if len(content) > 10:
                        data_list.append(f"{content}\t{label}\n")
            except Exception:
                continue

    # 保存新的 14 个类别列表
    with open(cfg.class_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_classes))
    print(f"类别表已保存至: {cfg.class_path}")

    random.shuffle(data_list)
    
    total = len(data_list)
    train_size = int(total * 0.8)
    dev_size = int(total * 0.1)
    
    train_data = data_list[:train_size]
    dev_data = data_list[train_size:train_size+dev_size]
    test_data = data_list[train_size+dev_size:]
    
    def save_file(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(data)
            
    save_file(cfg.train_path, train_data)
    save_file(cfg.dev_path, dev_data)
    save_file(cfg.test_path, test_data)
    
    print(f"数据处理完成！共 {total} 条数据。")

if __name__ == '__main__':
    preprocess_thucnews()
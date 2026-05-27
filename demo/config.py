# config.py
import torch
import os

class Config:
    def __init__(self):
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        
        # 原始数据路径 (官网下载解压后的文件夹路径)
        self.raw_data_path = os.path.join(self.root_path, r'D:\zhangn\xinwen\THUCNews')
        
        # 清洗后的数据存放路径
        self.data_dir = os.path.join(self.root_path, r'D:\zhangn\xinwen\data')
        self.train_path = os.path.join(self.data_dir, r'D:\zhangn\xinwen\train.txt')
        self.dev_path = os.path.join(self.data_dir, r'D:\zhangn\xinwen\dev.txt')
        self.test_path = os.path.join(self.data_dir, r'D:\zhangn\xinwen\test.txt')
        self.class_path = os.path.join(self.data_dir, r'D:\zhangn\xinwen\class.txt')
        
        self.bert_path = r'D:\zhangn\xinwen\bert' # 【关键】本地 BERT 模型路径elf.bert_path = os.path.join(self.bert_path = 'bert-base-chinese')
        
        # 结果保存
        self.output_dir = os.path.join(self.root_path, r'D:\zhangn\xinwen\output')
        self.save_path = os.path.join(self.output_dir, r'D:\zhangn\xinwen\bert_finetuned.ckpt')
        self.img_path = os.path.join(self.output_dir, r'D:\zhangn\xinwen\output\training_curve.png')
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 类别读取
        if os.path.exists(self.class_path):
            with open(self.class_path, 'r', encoding='utf-8') as f:
                self.class_list = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.class_list = [] # 预处理前可能为空

        self.num_classes = len(self.class_list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 训练超参数
        self.num_epochs = 5
        self.batch_size = 32
        self.pad_size = 128    # 截断长度，新闻一般较长，根据显存调整
        self.learning_rate = 2e-5
        self.require_improvement = 400 # 早停
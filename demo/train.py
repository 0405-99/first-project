# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import time
import os
from transformers import BertTokenizer, BertModel # 必须导入 BertModel
from torch.optim import AdamW  # 使用 PyTorch 官方推荐的 AdamW

from config import Config
from dataset import build_dataset, DatasetIterator
from utils import get_time_dif, plot_training_results

# ==============================================================================
# 【核心修改】直接在这里定义 Model 类，不再需要 model.py 文件
# ==============================================================================
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 加载本地的 Bert 预训练模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 开启微调
        for param in self.bert.parameters():
            param.requires_grad = True
        # 全连接分类层
        self.fc = nn.Linear(768, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入 [ids]
        mask = x[2]     # 掩码 [mask]
        # 传入 BERT
        outputs = self.bert(context, attention_mask=mask)
        # 获取 [CLS] token 的输出
        pooled = outputs.pooler_output 
        out = self.fc(pooled)
        return out
# ==============================================================================

def train(config):
    # 初始化 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    
    # 检查数据
    if not os.path.exists(config.train_path):
        print("【错误】未找到清洗后的数据。请先运行 'python preprocess.py'")
        return

    # 加载数据
    train_data, dev_data, test_data = build_dataset(config, tokenizer)
    train_iter = DatasetIterator(train_data, config.batch_size, config.device)
    dev_iter = DatasetIterator(dev_data, config.batch_size, config.device)
    test_iter = DatasetIterator(test_data, config.batch_size, config.device)

    # 实例化当前文件定义的 Model
    model = Model(config).to(config.device)
    model.train()
    
    # 参数优化设置
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    
    loss_history = []
    acc_history = []

    print("Start Training...")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                
                loss_history.append(loss.item())
                acc_history.append(dev_acc)
                
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
                
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag: break
    
    # 绘制图片
    plot_training_results(loss_history, acc_history, config.img_path)
    test(config, model, test_iter)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter)

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))

if __name__ == '__main__':
    config = Config()
    train(config)
# utils.py
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def plot_training_results(losses, accs, save_path):
    """
    绘制并保存训练过程的 Loss 和 Accuracy 曲线
    """
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Train Loss', color='#FF5733')
    plt.title('Training Loss')
    plt.xlabel('Steps (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(accs, label='Dev Accuracy', color='#33C1FF')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs (x100 steps)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(r'D:\zhangn\xinwen')
    plt.close()
    print(f"训练结果图已保存至: {save_path}")
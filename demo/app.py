# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from config import Config
import os
import pandas as pd
import time
from datetime import datetime

# ==============================================================================
# 1. 核心配置与模型定义 (无需外部 model.py)
# ==============================================================================

# 定义与 train.py 完全一致的模型结构
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs.pooler_output 
        out = self.fc(pooled)
        return out

# ==============================================================================
# 2. 页面初始化与自定义 CSS (打造科技感)
# ==============================================================================
st.set_page_config(
    page_title="THUCNews 智能中枢",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS 增强视觉效果
st.markdown("""
<style>
    /* 全局字体与背景微调 */
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00e676 !important;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 10px #00e676;
    }
    h2, h3 {
        color: #e0e0e0 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    /* 自定义指标卡片样式 */
    div[data-testid="metric-container"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"]:hover {
        border-color: #00e676;
    }
    /* 按钮特效 */
    .stButton>button {
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. 资源加载与 Session 管理
# ==============================================================================

# 初始化 Session State (用于存储历史记录)
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'total_runs' not in st.session_state:
    st.session_state['total_runs'] = 0

@st.cache_resource
def load_system():
    config = Config()
    config.device = torch.device('cpu') # 强制 CPU，保证稳定
    
    if not os.path.exists(config.save_path):
        return None, None, config
        
    try:
        tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        model = Model(config)
        model.load_state_dict(torch.load(config.save_path, map_location=config.device))
        model.to(config.device)
        model.eval()
        return tokenizer, model, config
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, None, config

def run_prediction(text, tokenizer, model, config):
    # 数据预处理
    token = tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    
    pad_size = config.pad_size
    mask = []
    token_ids = []
    
    if len(token) < pad_size:
        mask = [1] * len(token) + [0] * (pad_size - len(token))
        token_ids = tokenizer.convert_tokens_to_ids(token)
        token_ids += [0] * (pad_size - len(token))
    else:
        mask = [1] * pad_size
        token_ids = tokenizer.convert_tokens_to_ids(token[:pad_size])
        seq_len = pad_size
        
    x = torch.LongTensor([token_ids]).to(config.device)
    seq_len_tensor = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    
    # 推理
    start_time = time.time()
    with torch.no_grad():
        outputs = model((x, seq_len_tensor, mask))
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        prediction = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
    
    inference_time = time.time() - start_time
    label = config.class_list[prediction]
    
    return label, probs, inference_time

# ==============================================================================
# 4. 界面逻辑实现
# ==============================================================================

tokenizer, model, config = load_system()

# --- 侧边栏：控制台 ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=80)
    st.title("🛰️ 系统控制台")
    st.markdown("---")
    
    # 状态指示灯
    if model is not None:
        st.success("🟢 核心模型：在线")
        st.info(f"💾 模型路径：{config.save_path}")
    else:
        st.error("🔴 核心模型：离线")
        st.warning("请检查 `train.py` 是否运行完成")

    st.markdown("### 📊 运行统计")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("累计预测", st.session_state['total_runs'])
    col_s2.metric("类别总数", config.num_classes if config else 0)
    
    st.markdown("---")
    st.markdown("### 📥 数据导出")
    if st.session_state['history']:
        df_history = pd.DataFrame(st.session_state['history'])
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载历史记录 (CSV)",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.caption("暂无历史记录可下载")

# --- 主页面 ---
st.title("🧬 THUCNews 文本智能分类系统")
st.caption("Powered by BERT-Base-Chinese | Fine-Tuned System")

if model:
    # 输入区域
    with st.container():
        st.markdown("### 📝 待分析文本流")
        text_input = st.text_area(
            "请输入新闻内容 (支持长文本)", 
            height=150,
            placeholder="[系统准备就绪] 请在此输入需要分类的新闻文本..."
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            analyze = st.button("🚀 启动分析", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 清空历史", use_container_width=True):
                st.session_state['history'] = []
                st.rerun()

    # 分析逻辑
    if analyze and text_input:
        if not text_input.strip():
            st.warning("⚠️ 警告：检测到空输入流")
        else:
            with st.spinner("🔄 神经网络正在计算张量..."):
                label, probs, t_cost = run_prediction(text_input, tokenizer, model, config)
                
                # 更新状态
                st.session_state['total_runs'] += 1
                confidence = max(probs)
                
                # 记录历史
                st.session_state['history'].insert(0, {
                    "时间": datetime.now().strftime("%H:%M:%S"),
                    "文本摘要": text_input[:20] + "..." if len(text_input)>20 else text_input,
                    "预测类别": label,
                    "置信度": f"{confidence:.4f}",
                    "耗时(s)": f"{t_cost:.4f}"
                })

            # --- 结果展示区 ---
            st.markdown("---")
            st.subheader("🔍 分析报告 (Analysis Report)")
            
            # 第一行：核心指标
            c1, c2, c3 = st.columns(3)
            c1.metric("最终分类", label, delta="Match Found")
            c2.metric("置信度 (Confidence)", f"{confidence:.2%}", delta="High" if confidence > 0.8 else "Low")
            c3.metric("推理耗时", f"{t_cost:.4f} s")
            
            # 第二行：详细概率分布
            c_chart, c_detail = st.columns([2, 1])
            
            with c_chart:
                st.markdown("**概率分布雷达**")
                chart_data = pd.DataFrame({"Class": config.class_list, "Probability": probs})
                st.bar_chart(chart_data.set_index("Class"), color="#00e676")
                
            with c_detail:
                st.markdown("**Top 3 可能性**")
                # 排序获取前三
                top_indices = probs.argsort()[-3:][::-1]
                for idx in top_indices:
                    class_name = config.class_list[idx]
                    prob_val = probs[idx]
                    st.progress(float(prob_val), text=f"{class_name}: {prob_val:.2%}")

    # --- 历史记录区 ---
    if st.session_state['history']:
        st.markdown("---")
        with st.expander("📜 历史操作记录 (Session Logs)", expanded=True):
            st.dataframe(
                pd.DataFrame(st.session_state['history']),
                use_container_width=True,
                hide_index=True
            )

else:
    # 如果模型未找到的占位符
    st.info("💡 系统初始化指南：请确保已运行 `train.py` 并在 `output` 目录生成了模型权重。")
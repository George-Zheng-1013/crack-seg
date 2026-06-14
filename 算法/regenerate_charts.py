"""
重绘结题报告图表：使用 Box mAP 数据
- 图4-2: 训练 mAP@50 曲线对比 (Box mAP@50)
- 全局最优性能柱状图: 测试集 Box mAP 数据
- 精确率召回率曲线对比: Box precision/recall
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"D:\HP\OneDrive\Desktop\学校\课程\专业课\大数据综合工程设计"
RUNS_DIR = os.path.join(BASE_DIR, r"crack-seg\算法\runs")
OUTPUT_DIR = os.path.join(BASE_DIR, r"提交材料\图表")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 三组实验的训练阶段目录
SCHEMES = {
    '标准损失': {
        'dir': 'segment',
        'runs': ['train', 'train-2', 'train-3', 'train-4'],
        'color': '#4C72B0',
        'label': '标准损失'
    },
    '加权损失+余弦LR': {
        'dir': 'segment_coslr',
        'runs': ['train', 'train-2', 'train-3', 'train-4'],
        'color': '#DD8452',
        'label': '加权损失+余弦LR'
    },
    '加权损失': {
        'dir': 'segment_weighted',
        'runs': ['train', 'train-2', 'train-3', 'train-4'],
        'color': '#55A868',
        'label': '加权损失'
    }
}

def load_results(scheme_name):
    """加载某方案所有阶段的 results.csv 并拼接为全局 epoch 序列"""
    scheme = SCHEMES[scheme_name]
    dfs = []
    global_epoch = 0
    for run_name in scheme['runs']:
        csv_path = os.path.join(RUNS_DIR, scheme['dir'], run_name, 'results.csv')
        if not os.path.exists(csv_path):
            print(f"  跳过不存在的: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # 清理列名空格
        df.columns = df.columns.str.strip()
        df['global_epoch'] = range(global_epoch + 1, global_epoch + len(df) + 1)
        df['stage'] = run_name
        dfs.append(df)
        global_epoch += len(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ===== 加载所有数据 =====
print("加载训练数据...")
all_data = {}
for name in SCHEMES:
    df = load_results(name)
    all_data[name] = df
    print(f"  {name}: {len(df)} epochs")


# ===== 图1: 训练 mAP@50 曲线对比 (Box mAP) =====
print("\n绘制训练 mAP@50 曲线对比 (Box)...")
fig, ax = plt.subplots(figsize=(12, 6))

for name, scheme in SCHEMES.items():
    df = all_data[name]
    if df.empty:
        continue
    ax.plot(df['global_epoch'], df['metrics/mAP50(B)'],
            color=scheme['color'], label=scheme['label'], linewidth=1.5, alpha=0.85)
    # 标注最优点
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = df.loc[best_idx, 'global_epoch']
    best_val = df.loc[best_idx, 'metrics/mAP50(B)']
    ax.scatter([best_epoch], [best_val], color=scheme['color'], s=80, zorder=5,
               edgecolors='white', linewidths=1.5)
    ax.annotate(f'{best_val:.3f}\n(ep{best_epoch})',
                xy=(best_epoch, best_val),
                xytext=(0, 12), textcoords='offset points',
                fontsize=8, ha='center', color=scheme['color'], fontweight='bold')

# 标注训练阶段分隔线 (粗略估计每个阶段约16-32个epoch)
# 基于实际数据确定分隔位置
stage_boundaries = []
for name in SCHEMES:
    df = all_data[name]
    if df.empty:
        continue
    stages = df['stage'].unique()
    prev_stage = None
    for idx, row in df.iterrows():
        if prev_stage is not None and row['stage'] != prev_stage:
            if row['global_epoch'] not in stage_boundaries:
                stage_boundaries.append(row['global_epoch'] - 0.5)
        prev_stage = row['stage']

for b in stage_boundaries:
    ax.axvline(x=b, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

ax.set_xlabel('全局训练轮次 (Epoch)', fontsize=12)
ax.set_ylabel('mAP@50', fontsize=12)
ax.set_title('三组实验训练过程 mAP@50 曲线对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(1, max(df['global_epoch'].max() for df in all_data.values() if not df.empty))
ax.set_ylim(0, 0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '图4-2_训练mAP曲线对比.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  已保存: 图4-2_训练mAP曲线对比.png")


# ===== 图2: 全局最优性能柱状图 (测试集 Box mAP) =====
print("\n绘制全局最优性能柱状图 (测试集 Box mAP)...")

# 测试集数据 (来自 result.md)
test_data = {
    '标准损失': {'mAP@25': 0.450, 'mAP@50': 0.379},
    '加权损失+余弦LR': {'mAP@25': 0.457, 'mAP@50': 0.389},
    '加权损失': {'mAP@25': 0.452, 'mAP@50': 0.376},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# mAP@25
schemes = list(test_data.keys())
map25_vals = [test_data[s]['mAP@25'] for s in schemes]
map50_vals = [test_data[s]['mAP@50'] for s in schemes]
colors = [SCHEMES[s]['color'] for s in schemes]

bars1 = axes[0].bar(schemes, map25_vals, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars1, map25_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('mAP@25', fontsize=12)
axes[0].set_title('测试集 mAP@25', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 0.55)
axes[0].tick_params(axis='x', labelsize=9)
axes[0].grid(axis='y', alpha=0.3)

# mAP@50
bars2 = axes[1].bar(schemes, map50_vals, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, map50_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_ylabel('mAP@50', fontsize=12)
axes[1].set_title('测试集 mAP@50', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 0.35)
axes[1].tick_params(axis='x', labelsize=9)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '全局最优_性能柱状图.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  已保存: 全局最优_性能柱状图.png")


# ===== 图3: 各类别 mAP@50 对比 (测试集 Box mAP) =====
print("\n绘制各类别 mAP@50 测试集对比...")

per_class_data = {
    'Crack':       {'标准损失': 0.805, '加权损失+余弦LR': 0.824, '加权损失': 0.818},
    'Breakage':    {'标准损失': 0.450, '加权损失+余弦LR': 0.439, '加权损失': 0.435},
    'Comb':        {'标准损失': 0.240, '加权损失+余弦LR': 0.268, '加权损失': 0.263},
    'Hole':        {'标准损失': 0.475, '加权损失+余弦LR': 0.484, '加权损失': 0.436},
    'Reinforcement': {'标准损失': 0.437, '加权损失+余弦LR': 0.444, '加权损失': 0.442},
    'Seepage':     {'标准损失': 0.291, '加权损失+余弦LR': 0.283, '加权损失': 0.317},
}

fig, ax = plt.subplots(figsize=(12, 6))

classes = list(per_class_data.keys())
x = np.arange(len(classes))
width = 0.25

for i, (scheme_name, scheme) in enumerate(SCHEMES.items()):
    vals = [per_class_data[c][scheme_name] for c in classes]
    bars = ax.bar(x + i*width, vals, width, label=scheme['label'],
                  color=scheme['color'], edgecolor='white', linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=0)

ax.set_xlabel('缺陷类别', fontsize=12)
ax.set_ylabel('mAP@50', fontsize=12)
ax.set_title('各类别 mAP@25 测试集对比', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(classes, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '各类别mAP50测试集对比.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  已保存: 各类别mAP50测试集对比.png")


# ===== 图4: 精确率/召回率曲线对比 (Box) =====
print("\n绘制精确率/召回率曲线对比 (Box)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, scheme in SCHEMES.items():
    df = all_data[name]
    if df.empty:
        continue
    ax1.plot(df['global_epoch'], df['metrics/precision(B)'],
             color=scheme['color'], label=scheme['label'], linewidth=1.5, alpha=0.85)
    ax2.plot(df['global_epoch'], df['metrics/recall(B)'],
             color=scheme['color'], label=scheme['label'], linewidth=1.5, alpha=0.85)

ax1.set_xlabel('全局训练轮次 (Epoch)', fontsize=11)
ax1.set_ylabel('Precision (B)', fontsize=11)
ax1.set_title('精确率训练曲线', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xlim(1, max(df['global_epoch'].max() for df in all_data.values() if not df.empty))
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('全局训练轮次 (Epoch)', fontsize=11)
ax2.set_ylabel('Recall (B)', fontsize=11)
ax2.set_title('召回率训练曲线', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xlim(1, max(df['global_epoch'].max() for df in all_data.values() if not df.empty))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '精确率召回率曲线对比.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  已保存: 精确率召回率曲线对比.png")


# ===== 图5: Box mAP@50 全局训练曲线 (含阶段标记和最优点) =====
print("\n绘制 mAP@50 全局训练曲线 (含阶段标记)...")
fig, ax = plt.subplots(figsize=(14, 6))

for name, scheme in SCHEMES.items():
    df = all_data[name]
    if df.empty:
        continue
    ax.plot(df['global_epoch'], df['metrics/mAP50(B)'],
            color=scheme['color'], label=scheme['label'], linewidth=1.8, alpha=0.9)
    # 标注最优点
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = df.loc[best_idx, 'global_epoch']
    best_val = df.loc[best_idx, 'metrics/mAP50(B)']
    ax.scatter([best_epoch], [best_val], color=scheme['color'], s=100, zorder=5,
               marker='*', edgecolors='white', linewidths=1)
    ax.annotate(f'{best_val:.3f} (ep{int(best_epoch)})',
                xy=(best_epoch, best_val),
                xytext=(10, 8), textcoords='offset points',
                fontsize=9, color=scheme['color'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=scheme['color'], lw=1))

# 绘制阶段分隔背景色
max_ep = max(df['global_epoch'].max() for df in all_data.values() if not df.empty)
stage_colors = ['#f0f0f0', '#e8e8e8']
# 简单的分段：基于第一个方案的阶段信息
first_df = list(all_data.values())[0]
if not first_df.empty:
    stage_changes = []
    prev_stage = None
    for idx, row in first_df.iterrows():
        if prev_stage is not None and row['stage'] != prev_stage:
            stage_changes.append(row['global_epoch'] - 0.5)
        prev_stage = row['stage']
    
    boundaries = [0] + stage_changes + [max_ep + 1]
    for i in range(len(boundaries) - 1):
        ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.15,
                   color=stage_colors[i % 2])

ax.set_xlabel('全局训练轮次 (Epoch)', fontsize=12)
ax.set_ylabel('mAP@50', fontsize=12)
ax.set_title('mAP@50 全局训练曲线（含训练阶段标记与最优点标注）', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(1, max_ep)
ax.set_ylim(0, 0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Box_mAP50_全局训练曲线.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  已保存: Box_mAP50_全局训练曲线.png")


# ===== 汇总 =====
print("\n===== 图表重绘完成 =====")
print(f"输出目录: {OUTPUT_DIR}")
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.png'):
        fpath = os.path.join(OUTPUT_DIR, f)
        fsize = os.path.getsize(fpath) / 1024
        print(f"  {f}: {fsize:.1f} KB")

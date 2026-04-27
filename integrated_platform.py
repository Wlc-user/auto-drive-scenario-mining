"""
自动驾驶接管行为根因分析系统
功能：从多维数据中自动识别接管事件，分析根本原因，生成可视化报告
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据模拟 ====================
def generate_rich_driving_log(duration_s=1800, fs=10, seed=42):
    """
    生成带接管标记的丰富行车日志
    字段：时间戳、自车速度、加速度、方向盘转角、前车距离、相对速度、TTC、车道偏离量、OD目标类型、接管标记
    """
    np.random.seed(seed)
    n = duration_s * fs
    t = np.arange(n) / fs
    
    # 基础数据
    speed = 30 + 15 * np.sin(2*np.pi * t/300)
    speed += np.random.randn(n) * 2
    speed = np.clip(speed, 5, 70)
    
    acc = np.gradient(speed, 1/fs)
    
    steering = 2 * np.sin(2*np.pi * t/45) + np.random.randn(n) * 1.5
    
    # 前车距离（随机变化）
    front_distance = 20 + 10 * np.sin(2*np.pi * t/90) + np.random.randn(n) * 3
    front_distance = np.clip(front_distance, 2, 50)
    
    # 相对速度
    rel_speed = -3 + np.random.randn(n) * 2
    
    # TTC (Time To Collision)
    ttc = np.where(rel_speed < -0.5, front_distance / abs(rel_speed), 99)
    ttc = np.clip(ttc, 0.5, 99)
    
    # 车道偏离量
    lane_deviation = np.random.randn(n) * 0.2
    lane_deviation = np.clip(lane_deviation, -0.8, 0.8)
    
    # 目标物类型
    obj_types = np.random.choice(['车辆', '行人', '自行车', '无'], size=n, p=[0.4, 0.2, 0.1, 0.3])
    
    # 接管标记（初始为0）
    takeover = np.zeros(n)
    
    # 注入接管事件（5个不同原因的事件）
    event_configs = [
        {'time': 200, 'cause': '前向碰撞风险', 'setup': lambda idx: setup_ttc_risk(idx, ttc, front_distance, rel_speed)},
        {'time': 500, 'cause': '车道保持异常', 'setup': lambda idx: setup_lane_risk(idx, lane_deviation)},
        {'time': 800, 'cause': '紧急制动触发', 'setup': lambda idx: setup_emergency_brake(idx, acc, speed)},
        {'time': 1100, 'cause': '传感器遮挡', 'setup': lambda idx: setup_sensor_block(idx, obj_types)},
        {'time': 1400, 'cause': '前向碰撞风险', 'setup': lambda idx: setup_ttc_risk(idx, ttc, front_distance, rel_speed)},
    ]
    
    for config in event_configs:
        idx = int(config['time'] * fs)
        config['setup'](idx)
        takeover[idx:idx+5] = 1  # 接管标记持续0.5秒
    
    df = pd.DataFrame({
        '时间戳': t,
        '自车速度': speed,
        '加速度': acc,
        '方向盘转角': steering,
        '前车距离': front_distance,
        '相对速度': rel_speed,
        'TTC': ttc,
        '车道偏离量': lane_deviation,
        'OD目标类型': obj_types,
        '接管标记': takeover
    })
    
    return df

def setup_ttc_risk(idx, ttc, front_distance, rel_speed):
    """设置前向碰撞风险场景"""
    ttc[idx-50:idx] = np.linspace(10, 1.5, 50)
    front_distance[idx-50:idx] = np.linspace(30, 5, 50)

def setup_lane_risk(idx, lane_deviation):
    """设置车道偏离风险场景"""
    lane_deviation[idx-30:idx] = np.linspace(0.1, 0.7, 30)

def setup_emergency_brake(idx, acc, speed):
    """设置紧急制动场景"""
    acc[idx-20:idx] = np.linspace(-2, -7, 20)

def setup_sensor_block(idx, obj_types):
    """设置传感器遮挡（目标物变为'无'）"""
    obj_types[idx-40:idx] = '无'

# ==================== 2. 接管事件检测与特征提取 ====================
@dataclass
class TakeoverEvent:
    """接管事件"""
    idx: int
    time: float
    features: Dict
    root_cause: str

class TakeoverAnalyzer:
    """接管行为分析器"""
    
    def __init__(self, fs=10):
        self.fs = fs
        self.pre_window = int(5 * fs)  # 接管前5秒
    
    def detect_takeovers(self, df):
        """检测接管事件"""
        takeover_indices = df[df['接管标记'] == 1].index
        if len(takeover_indices) == 0:
            return []
        
        # 合并连续的接管标记
        events = []
        i = 0
        while i < len(takeover_indices):
            start_idx = takeover_indices[i]
            # 找连续区间的结束
            j = i
            while j + 1 < len(takeover_indices) and takeover_indices[j+1] - takeover_indices[j] == 1:
                j += 1
            events.append(start_idx)
            i = j + 1
        
        return events
    
    def extract_features(self, df, event_idx):
        """提取接管前特征"""
        start = max(0, event_idx - self.pre_window)
        segment = df.iloc[start:event_idx+1]
        
        features = {
            'min_TTC': segment['TTC'].min(),
            'TTC_下降率': (segment['TTC'].iloc[-10] - segment['TTC'].iloc[0]) / (len(segment)-10) if len(segment) > 10 else 0,
            'max_lane_deviation': abs(segment['车道偏离量']).max(),
            'lane_deviation_std': segment['车道偏离量'].std(),
            'max_deceleration': segment['加速度'].min(),
            'speed_at_takeover': segment['自车速度'].iloc[-1],
            'front_distance_min': segment['前车距离'].min(),
            'front_distance_at_takeover': segment['前车距离'].iloc[-1],
            'dominant_obj_type': segment['OD目标类型'].mode().iloc[0] if len(segment['OD目标类型'].mode()) > 0 else '无',
            'obj_changes': (segment['OD目标类型'] != segment['OD目标类型'].shift()).sum(),
        }
        
        return features
    
    def classify_root_cause(self, features):
        """基于规则的根本原因分类"""
        if features['min_TTC'] < 2.0:
            return '前向碰撞风险'
        elif features['max_lane_deviation'] > 0.5:
            return '车道保持异常'
        elif features['max_deceleration'] < -5.0:
            return '紧急制动触发'
        elif features['dominant_obj_type'] == '无' and features['front_distance_at_takeover'] < 15:
            return '传感器遮挡/目标丢失'
        elif features['obj_changes'] > 10:
            return '感知不稳定'
        else:
            return '驾驶员主动介入'

# ==================== 3. 可视化 ====================
def plot_root_cause_distribution(causes, save_path='root_cause_distribution.png'):
    """绘制根因分布饼图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 饼图
    cause_counts = Counter(causes)
    colors = sns.color_palette('Set2', len(cause_counts))
    wedges, texts, autotexts = axes[0].pie(
        cause_counts.values(), 
        labels=cause_counts.keys(), 
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05]*len(cause_counts)
    )
    axes[0].set_title('接管根因分布', fontsize=14, fontweight='bold')
    
    # 柱状图
    cause_df = pd.DataFrame(list(cause_counts.items()), columns=['原因', '次数'])
    cause_df = cause_df.sort_values('次数', ascending=False)
    bars = axes[1].barh(cause_df['原因'], cause_df['次数'], color=colors)
    axes[1].set_xlabel('接管次数')
    axes[1].set_title('接管原因排序', fontsize=14, fontweight='bold')
    # 添加数值标签
    for bar, val in zip(bars, cause_df['次数']):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 根因分布图已保存: {save_path}')

def plot_feature_comparison(events_dict, save_path='feature_comparison.png'):
    """对比不同根因下的特征分布"""
    # 整理数据
    records = []
    for cause, events in events_dict.items():
        for feat in events:
            feat['根因'] = cause
            records.append(feat)
    
    df = pd.DataFrame(records)
    
    key_features = ['min_TTC', 'max_lane_deviation', 'max_deceleration', 'front_distance_at_takeover']
    labels = ['最小TTC', '最大车道偏离', '最大减速度', '接管时前车距离']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (feat, label) in enumerate(zip(key_features, labels)):
        # 只取有数据的根因类别
        data_by_cause = [df[df['根因']==c][feat].values for c in df['根因'].unique() if len(df[df['根因']==c]) > 0]
        labels_by_cause = [c for c in df['根因'].unique() if len(df[df['根因']==c]) > 0]
        
        bp = axes[i].boxplot(data_by_cause, labels=labels_by_cause, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette('Set2', len(labels_by_cause))):
            patch.set_facecolor(color)
        axes[i].set_title(label, fontsize=13, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('不同根因接管事件的特征对比', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 特征对比图已保存: {save_path}')

def plot_takeover_timeline(events, save_path='takeover_timeline.png'):
    """绘制接管事件时间线"""
    if not events:
        return
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    causes = [e.root_cause for e in events]
    times = [e.time for e in events]
    colors = {'前向碰撞风险': '#FF6B6B', '车道保持异常': '#4ECDC4', '紧急制动触发': '#45B7D1', 
              '传感器遮挡/目标丢失': '#96CEB4', '驾驶员主动介入': '#D4A5A5', '感知不稳定': '#9B59B6'}
    
    for i, (t, c) in enumerate(zip(times, causes)):
        color = colors.get(c, '#888888')
        ax.axvline(x=t, color=color, linestyle='-', linewidth=2, alpha=0.7)
        ax.text(t, 0.5 + (i % 3)*0.15, f'{c}\n({t:.0f}s)', ha='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_title('接管事件时间线', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 接管时间线已保存: {save_path}')

# ==================== 4. 主程序 ====================
if __name__ == '__main__':
    print('🔬 自动驾驶接管行为根因分析系统启动...\n')
    
    # 生成数据
    print('📡 生成模拟行车日志（含接管标记）...')
    df = generate_rich_driving_log(duration_s=1800, fs=10)
    print(f'   生成 {len(df)} 条记录，{df["接管标记"].sum():.0f} 个接管标记点')
    
    # 分析接管
    print('🔍 分析接管事件...')
    analyzer = TakeoverAnalyzer(fs=10)
    event_indices = analyzer.detect_takeovers(df)
    
    events = []
    for idx in event_indices:
        features = analyzer.extract_features(df, idx)
        root_cause = analyzer.classify_root_cause(features)
        event = TakeoverEvent(idx=idx, time=df['时间戳'].iloc[idx], features=features, root_cause=root_cause)
        events.append(event)
    
    print(f'   发现 {len(events)} 次接管事件')
    
    # 打印分析报告
    print('\n' + '='*60)
    print('📊 接管根因分析报告')
    print('='*60)
    
    events_by_cause = {}
    for event in events:
        if event.root_cause not in events_by_cause:
            events_by_cause[event.root_cause] = []
        events_by_cause[event.root_cause].append(event.features)
        
        print(f'\n接管 #{events.index(event)+1}')
        print(f'  时间: {event.time:.1f}s')
        print(f'  根因: {event.root_cause}')
        print(f'  最小TTC: {event.features["min_TTC"]:.2f}s')
        print(f'  最大车道偏离: {event.features["max_lane_deviation"]:.2f}m')
        print(f'  最大减速度: {event.features["max_deceleration"]:.1f} m/s²')
        print(f'  主要目标类型: {event.features["dominant_obj_type"]}')
    
    # 统计
    cause_counts = Counter([e.root_cause for e in events])
    print(f'\n--- 根因分布 ---')
    for cause, count in cause_counts.most_common():
        print(f'  {cause}: {count}次 ({count/len(events)*100:.1f}%)')
    print('='*60)
    
    # 生成可视化
    print('\n🎨 生成可视化报告...')
    os.makedirs('reports', exist_ok=True)
    plot_root_cause_distribution([e.root_cause for e in events], 'reports/root_cause_distribution.png')
    plot_feature_comparison(events_by_cause, 'reports/feature_comparison.png')
    plot_takeover_timeline(events, 'reports/takeover_timeline.png')
    
    print('\n✅ 接管根因分析完成！所有报告图片保存在 reports/ 目录')
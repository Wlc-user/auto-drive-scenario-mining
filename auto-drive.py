"""
急刹车场景自动化挖掘系统
功能：从行车日志中检测急刹车事件，提取场景片段，生成统计报告
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据模拟 ====================
def generate_driving_log(duration_s=600, fs=10, seed=42):
    """生成模拟行车日志"""
    np.random.seed(seed)
    n = duration_s * fs
    t = np.arange(n) / fs
    
    # 基础车速变化（模拟城市道路）
    base_speed = 30 + 10 * np.sin(2*np.pi * t/120)  # 2分钟周期波动
    base_speed += np.random.randn(n) * 3  # 随机波动
    base_speed = np.clip(base_speed, 0, 60)
    
    # 加速度（通过速度差分计算）
    acc = np.gradient(base_speed, 1/fs)
    
    # 方向盘角度
    steering = 3 * np.sin(2*np.pi * t/30) + np.random.randn(n) * 2
    
    # 在第一段中，故意插入急刹车事件
    event_times = [60, 180, 300, 420, 500]  # 事件发生时间（秒）
    for et in event_times:
        idx = int(et * fs)
        # 插入一段急减速
        if idx + 10 < n:
            acc[idx:idx+5] = np.linspace(-2, -6, 5)
            acc[idx+5:idx+8] = -6 + np.random.randn(3) * 0.5
            acc[idx+8:idx+10] = np.linspace(-6, -2, 2)
    
    # 通过加速度反推速度（保证连续性）
    speed = np.zeros(n)
    speed[0] = base_speed[0]
    for i in range(1, n):
        speed[i] = speed[i-1] + acc[i] * (1/fs)
        speed[i] = max(0, min(70, speed[i]))
    
    df = pd.DataFrame({
        '时间戳': t,
        '车速': speed,
        '加速度': acc,
        '方向盘角度': steering
    })
    
    return df

# ==================== 2. 事件检测核心算法 ====================
@dataclass
class BrakingEvent:
    """急刹车事件数据结构"""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration_s: float
    max_deceleration: float
    initial_speed: float
    speed_drop: float

class HardBrakingDetector:
    """急刹车检测器 - 基于状态机"""
    
    def __init__(self, threshold_mps2=-4.0, min_duration_s=0.5, fs=10):
        self.threshold = threshold_mps2
        self.min_samples = int(min_duration_s * fs)
        self.fs = fs
    
    def detect(self, df: pd.DataFrame) -> List[BrakingEvent]:
        """检测所有急刹车事件"""
        events = []
        in_event = False
        event_start = None
        acc_values = df['加速度'].values
        
        for i in range(len(acc_values)):
            if acc_values[i] < self.threshold and not in_event:
                # 进入急刹车状态
                in_event = True
                event_start = i
            elif acc_values[i] >= self.threshold and in_event:
                # 退出急刹车状态
                duration_samples = i - event_start
                if duration_samples >= self.min_samples:
                    segment = acc_values[event_start:i]
                    initial_speed = df['车速'].iloc[event_start]
                    final_speed = df['车速'].iloc[i-1] if i > 0 else initial_speed
                    
                    events.append(BrakingEvent(
                        start_idx=event_start,
                        end_idx=i - 1,
                        start_time=df['时间戳'].iloc[event_start],
                        end_time=df['时间戳'].iloc[i-1],
                        duration_s=duration_samples / self.fs,
                        max_deceleration=segment.min(),
                        initial_speed=initial_speed,
                        speed_drop=initial_speed - final_speed
                    ))
                in_event = False
        
        # 处理事件持续到数据末尾的情况
        if in_event:
            duration_samples = len(acc_values) - event_start
            if duration_samples >= self.min_samples:
                segment = acc_values[event_start:]
                initial_speed = df['车速'].iloc[event_start]
                final_speed = df['车速'].iloc[-1]
                events.append(BrakingEvent(
                    start_idx=event_start,
                    end_idx=len(acc_values) - 1,
                    start_time=df['时间戳'].iloc[event_start],
                    end_time=df['时间戳'].iloc[-1],
                    duration_s=duration_samples / self.fs,
                    max_deceleration=segment.min(),
                    initial_speed=initial_speed,
                    speed_drop=initial_speed - final_speed
                ))
        
        return events
    
    def extract_scene(self, df, event, pre_s=3, post_s=5):
        """提取事件前后数据片段"""
        pre_samples = int(pre_s * self.fs)
        post_samples = int(post_s * self.fs)
        
        start = max(0, event.start_idx - pre_samples)
        end = min(len(df), event.end_idx + post_samples)
        
        scene = df.iloc[start:end].copy()
        scene['相对时间'] = scene['时间戳'] - event.start_time
        scene['是否急刹'] = False
        # 标注急刹区间
        event_start_rel = event.start_time - event.start_time  # 0
        scene.loc[(scene['相对时间'] >= 0) & (scene['相对时间'] <= event.duration_s), '是否急刹'] = True
        
        return scene

# ==================== 3. 可视化 ====================
def plot_events_overview(df, events, save_path='event_overview.png'):
    """绘制全时段数据及事件标注"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    # 车速
    axes[0].plot(df['时间戳'], df['车速'], 'b-', linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel('车速 (km/h)')
    axes[0].grid(True, alpha=0.3)
    
    # 加速度
    axes[1].plot(df['时间戳'], df['加速度'], 'g-', linewidth=0.8, alpha=0.7)
    axes[1].axhline(y=-4.0, color='r', linestyle='--', alpha=0.5, label='急刹阈值')
    axes[1].set_ylabel('加速度 (m/s²)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 标注事件区域
    for event in events:
        for ax in axes[:2]:
            ax.axvspan(event.start_time, event.end_time, alpha=0.2, color='red')
    
    # 方向盘角度
    axes[2].plot(df['时间戳'], df['方向盘角度'], 'orange', linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel('方向盘角度 (°)')
    axes[2].set_xlabel('时间 (秒)')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('行车日志全景视图 - 红色区域为检测到的急刹车事件', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 全景图已保存: {save_path}')

def plot_event_details(scenes, events, save_dir='scenes'):
    """绘制每个急刹事件的细节图"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (scene, event) in enumerate(zip(scenes, events)):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # 速度曲线
        axes[0].plot(scene['相对时间'], scene['车速'], 'b-', linewidth=2)
        # 标注急刹区间
        mask = scene['是否急刹']
        if mask.any():
            axes[0].fill_between(scene['相对时间'][mask], 0, scene['车速'][mask].max(), 
                                  alpha=0.3, color='red', label='急刹区间')
        axes[0].set_ylabel('车速 (km/h)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 加速度曲线
        colors = ['red' if x else 'green' for x in scene['是否急刹']]
        axes[1].scatter(scene['相对时间'], scene['加速度'], c=colors, s=20, alpha=0.6)
        axes[1].axhline(y=-4.0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('加速度 (m/s²)')
        axes[1].set_xlabel('相对时间 (秒)')
        axes[1].grid(True, alpha=0.3)
        
        # 事件信息
        info_text = f"初始速度: {event.initial_speed:.1f} km/h\n速度降幅: {event.speed_drop:.1f} km/h\n最大减速度: {event.max_deceleration:.1f} m/s²"
        axes[0].text(0.02, 0.95, info_text, transform=axes[0].transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'急刹车事件 #{i+1} - 持续时间 {event.duration_s:.1f}秒', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/event_{i+1:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f'✅ {len(events)}个事件细节图已保存到 {save_dir}/')

# ==================== 4. 统计报告 ====================
def generate_report(df, events):
    """生成统计报告"""
    print('\n' + '='*60)
    print('📊 急刹车场景挖掘报告')
    print('='*60)
    print(f'数据总时长: {df["时间戳"].iloc[-1]:.0f} 秒')
    print(f'检测到急刹车事件: {len(events)} 次')
    print(f'平均每{df["时间戳"].iloc[-1]/60/len(events) if events else 0:.1f}分钟一次')
    
    if events:
        durations = [e.duration_s for e in events]
        decels = [e.max_deceleration for e in events]
        speeds = [e.initial_speed for e in events]
        
        print(f'\n--- 事件特征统计 ---')
        print(f'持续时间: 平均 {np.mean(durations):.2f}s, 范围 [{min(durations):.2f}, {max(durations):.2f}]s')
        print(f'最大减速度: 平均 {np.mean(decels):.2f} m/s², 最剧烈 {min(decels):.2f} m/s²')
        print(f'触发时车速: 平均 {np.mean(speeds):.1f} km/h')
        
        # 速度分布
        speed_bins = [0, 20, 40, 60, 100]
        speed_labels = ['0-20', '20-40', '40-60', '60+']
        speed_dist = pd.cut(speeds, bins=speed_bins, labels=speed_labels)
        print(f'\n--- 触发速度分布 ---')
        for label in speed_labels:
            count = (speed_dist == label).sum()
            print(f'  {label} km/h: {count} 次 ({count/len(events)*100:.1f}%)')
    
    print('='*60)


# ==================== 主程序 ====================
if __name__ == '__main__':
    print('🚗 自动驾驶急刹车场景挖掘系统启动...\n')
    
    # 1. 生成模拟数据
    print('📡 生成模拟行车日志...')
    df = generate_driving_log(duration_s=600, fs=10)
    print(f'   生成 {len(df)} 条数据记录')
    
    # 2. 初始化检测器并检测事件
    print('🔍 检测急刹车事件...')
    detector = HardBrakingDetector(threshold_mps2=-4.0, min_duration_s=0.5, fs=10)
    events = detector.detect(df)
    print(f'   发现 {len(events)} 个急刹车事件')
    
    # 3. 提取场景片段
    print('📦 提取事件场景片段...')
    scenes = [detector.extract_scene(df, e) for e in events]
    
    # 4. 生成统计报告
    generate_report(df, events)
    
    # 5. 可视化
    print('\n🎨 生成可视化图表...')
    plot_events_overview(df, events)
    plot_event_details(scenes, events)
    
    # 6. 导出场景数据
    print('💾 导出场景数据...')
    os.makedirs('scenes', exist_ok=True)
    for i, scene in enumerate(scenes):
        scene.to_csv(f'scenes/event_{i+1:02d}_data.csv', index=False)
    print(f'✅ {len(scenes)}个场景CSV文件已保存\n')
    
    print('✅ 急刹车场景挖掘完成！')
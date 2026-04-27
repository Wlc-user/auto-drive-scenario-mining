"""
自动驾驶场景数据挖掘与分析平台
整合：急刹车检测 + 接管根因分析 + 相似场景搜索
支持大数据量处理（分块读取、降采样、并行处理）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import Counter
import os
import warnings
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第一部分：数据结构定义
# ============================================================

@dataclass
class BrakingEvent:
    """急刹车事件"""
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration_s: float
    max_deceleration: float
    initial_speed: float
    speed_drop: float


@dataclass
class TakeoverEvent:
    """接管事件"""
    idx: int
    time: float
    features: Dict
    root_cause: str


@dataclass
class SceneInfo:
    """统一场景信息"""
    scene_id: int
    scene_type: str
    start_time: float
    end_time: float
    data: pd.DataFrame
    braking_events: List[BrakingEvent]
    takeover_events: List[TakeoverEvent]
    feature_vector: Optional[np.ndarray] = None


# ============================================================
# 第二部分：大数据工具类
# ============================================================

class BigDataProcessor:
    """大数据处理工具"""
    
    @staticmethod
    def load_large_csv(file_path: str, chunk_size: int = 100000, 
                       use_cols: List[str] = None) -> pd.DataFrame:
        """分块读取大型CSV文件，内存友好"""
        print(f'📂 分块读取文件: {file_path}')
        chunks = []
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, 
                                               usecols=use_cols)):
            # 数据清洗
            chunk = chunk.dropna()
            if '车速' in chunk.columns:
                chunk = chunk[(chunk['车速'] >= 0) & (chunk['车速'] <= 120)]
            if '加速度' in chunk.columns:
                chunk = chunk[(chunk['加速度'] >= -15) & (chunk['加速度'] <= 15)]
            
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if (i + 1) % 10 == 0:
                print(f'   已读取 {total_rows:,} 行...')
        
        result = pd.concat(chunks, ignore_index=True)
        print(f'✅ 读取完成: {total_rows:,} 行, 内存占用 {result.memory_usage(deep=True).sum()/1024/1024:.1f} MB')
        return result
    
    @staticmethod
    def downsample(df: pd.DataFrame, fs: int = 10, target_fs: int = 1) -> pd.DataFrame:
        """降采样：10Hz → 1Hz，减少90%数据量"""
        if '时间戳' not in df.columns:
            return df
        
        df = df.copy()
        df['时间戳'] = pd.to_datetime(df['时间戳'], unit='s')
        df = df.set_index('时间戳')
        
        # 降采样到目标频率
        rule = f'{int(1000/target_fs)}ms'
        df_downsampled = df.resample(rule).mean()
        df_downsampled = df_downsampled.dropna()
        
        print(f'📉 降采样: {len(df):,} → {len(df_downsampled):,} 行 (节省 {(1-len(df_downsampled)/len(df))*100:.1f}%)')
        return df_downsampled.reset_index()
    
    @staticmethod
    def batch_process_folder(folder_path: str, processor_func, 
                             file_pattern: str = '*.csv', **kwargs) -> List:
        """批量处理文件夹内所有日志文件"""
        import glob
        
        all_results = []
        files = glob.glob(os.path.join(folder_path, file_pattern))
        print(f'📁 批量处理 {len(files)} 个文件...')
        
        for i, file_path in enumerate(files):
            print(f'  [{i+1}/{len(files)}] 处理: {os.path.basename(file_path)}')
            try:
                result = processor_func(file_path, **kwargs)
                all_results.append(result)
            except Exception as e:
                print(f'    ⚠️ 处理失败: {e}')
        
        return all_results


# ============================================================
# 第三部分：场景数据生成器
# ============================================================

class ScenarioDataGenerator:
    """生成模拟的行车场景数据"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_log_with_events(self, duration_s: int = 1800, fs: int = 10) -> pd.DataFrame:
        """生成包含急刹车和接管事件的完整行车日志"""
        n = duration_s * fs
        t = np.arange(n) / fs
        
        # 基础行车数据
        speed = 30 + 15 * np.sin(2 * np.pi * t / 300) + np.random.randn(n) * 2
        speed = np.clip(speed, 5, 70)
        
        acc = np.gradient(speed, 1 / fs)
        
        steering = 2 * np.sin(2 * np.pi * t / 45) + np.random.randn(n) * 1.5
        
        # 前车数据
        front_distance = 20 + 10 * np.sin(2 * np.pi * t / 90) + np.random.randn(n) * 3
        front_distance = np.clip(front_distance, 2, 50)
        
        rel_speed = -3 + np.random.randn(n) * 2
        ttc = np.where(rel_speed < -0.5, front_distance / abs(rel_speed), 99)
        ttc = np.clip(ttc, 0.5, 99)
        
        # 车道数据
        lane_deviation = np.random.randn(n) * 0.2
        lane_deviation = np.clip(lane_deviation, -0.8, 0.8)
        
        # 目标物类型
        obj_types = np.random.choice(['车辆', '行人', '自行车', '无'], 
                                     size=n, p=[0.4, 0.2, 0.1, 0.3])
        
        # 接管标记
        takeover = np.zeros(n)
        
        # 注入6个复杂事件
        self._inject_events(n, fs, acc, speed, front_distance, rel_speed, 
                           ttc, lane_deviation, obj_types, takeover)
        
        df = pd.DataFrame({
            '时间戳': t,
            '车速': speed,
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
    
    def _inject_events(self, n, fs, acc, speed, front_distance, rel_speed, 
                       ttc, lane_deviation, obj_types, takeover):
        """注入多种场景事件"""
        
        # 事件1: 急刹车 + 前向碰撞风险
        idx = int(200 * fs)
        acc[idx:idx+20] = np.linspace(-2, -7, 20)
        ttc[idx-50:idx] = np.linspace(10, 1.2, 50)
        front_distance[idx-50:idx] = np.linspace(30, 4, 50)
        takeover[idx+20:idx+25] = 1
        
        # 事件2: 车道偏离
        idx = int(500 * fs)
        lane_deviation[idx-30:idx+10] = np.linspace(-0.1, 0.75, 40)
        takeover[idx:idx+5] = 1
        
        # 事件3: 纯急刹车
        idx = int(800 * fs)
        acc[idx:idx+15] = np.linspace(-1, -6, 15)
        
        # 事件4: 传感器遮挡
        idx = int(1100 * fs)
        obj_types[idx-40:idx] = '无'
        front_distance[idx-40:idx] = np.linspace(20, 3, 40)
        takeover[idx:idx+5] = 1
        
        # 事件5: 另一个急刹车
        idx = int(1400 * fs)
        acc[idx:idx+12] = np.linspace(-2, -5.5, 12)
        
        # 事件6: 目标丢失
        idx = int(1600 * fs)
        obj_types[idx-30:idx] = '无'
        takeover[idx:idx+5] = 1
    
    def generate_scene_library(self, n_scenes: int = 200, 
                               scene_length_s: int = 8, fs: int = 10) -> List[pd.DataFrame]:
        """生成场景片段库"""
        scenes = []
        fixed_length = scene_length_s * fs
        
        for i in range(n_scenes):
            t = np.arange(fixed_length) / fs
            scene_type = np.random.choice(['normal', 'braking', 'acceleration', 'turning'],
                                          p=[0.4, 0.25, 0.15, 0.2])
            
            if scene_type == 'normal':
                speed = 30 + np.random.randn(fixed_length) * 3
                acc = np.random.randn(fixed_length) * 1
                steering = np.random.randn(fixed_length) * 2
            
            elif scene_type == 'braking':
                speed = np.linspace(45, 8, fixed_length) + np.random.randn(fixed_length) * 2
                acc = np.full(fixed_length, -4.5) + np.random.randn(fixed_length) * 1.5
                steering = np.random.randn(fixed_length) * 3
            
            elif scene_type == 'acceleration':
                speed = np.linspace(8, 55, fixed_length) + np.random.randn(fixed_length) * 2
                acc = np.full(fixed_length, 3.5) + np.random.randn(fixed_length) * 1
                steering = np.random.randn(fixed_length) * 1
            
            else:  # turning
                speed = 25 + np.random.randn(fixed_length) * 2
                acc = np.random.randn(fixed_length) * 1
                steering = np.sin(2 * np.pi * t / 2) * 20 + np.random.randn(fixed_length) * 3
            
            scene_df = pd.DataFrame({
                '时间': t,
                '车速': np.clip(speed, 0, 80),
                '加速度': acc,
                '方向盘转角': np.clip(steering, -40, 40),
                '场景类型': scene_type
            })
            scenes.append(scene_df)
        
        return scenes


# ============================================================
# 第四部分：检测与分析引擎
# ============================================================

class HardBrakingDetector:
    """急刹车检测器"""
    
    def __init__(self, threshold_mps2: float = -4.0, min_duration_s: float = 0.5, fs: int = 10):
        self.threshold = threshold_mps2
        self.min_samples = int(min_duration_s * fs)
        self.fs = fs
    
    def detect(self, df: pd.DataFrame) -> List[BrakingEvent]:
        """检测急刹车事件（状态机算法）"""
        events = []
        in_event = False
        event_start = None
        acc_values = df['加速度'].values
        
        for i in range(len(acc_values)):
            if acc_values[i] < self.threshold and not in_event:
                in_event = True
                event_start = i
            elif acc_values[i] >= self.threshold and in_event:
                in_event = False
                if i - event_start >= self.min_samples:
                    events.append(self._create_event(df, event_start, i))
        
        # 处理末尾事件
        if in_event and len(acc_values) - event_start >= self.min_samples:
            events.append(self._create_event(df, event_start, len(acc_values)))
        
        return events
    
    def _create_event(self, df, start, end):
        """创建事件对象"""
        segment = df['加速度'].iloc[start:end]
        initial_speed = df['车速'].iloc[start]
        final_speed = df['车速'].iloc[min(end, len(df)-1)]
        
        return BrakingEvent(
            start_idx=start,
            end_idx=end - 1,
            start_time=df['时间戳'].iloc[start],
            end_time=df['时间戳'].iloc[min(end-1, len(df)-1)],
            duration_s=(end - start) / self.fs,
            max_deceleration=segment.min(),
            initial_speed=initial_speed,
            speed_drop=initial_speed - final_speed
        )
    
    def detect_in_chunks(self, file_path: str, chunk_size: int = 100000) -> List[BrakingEvent]:
        """分块处理大文件（内存友好）"""
        all_events = []
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_events = self.detect(chunk)
            all_events.extend(chunk_events)
        
        return all_events


class TakeoverAnalyzer:
    """接管行为分析器"""
    
    def __init__(self, fs: int = 10):
        self.fs = fs
        self.pre_window = int(5 * fs)
    
    def detect_takeovers(self, df: pd.DataFrame) -> List[int]:
        """检测接管事件的起始索引"""
        takeover_indices = df[df['接管标记'] == 1].index
        if len(takeover_indices) == 0:
            return []
        
        events = []
        i = 0
        while i < len(takeover_indices):
            start = takeover_indices[i]
            j = i
            while j + 1 < len(takeover_indices) and takeover_indices[j+1] - takeover_indices[j] == 1:
                j += 1
            events.append(start)
            i = j + 1
        
        return events
    
    def extract_features(self, df: pd.DataFrame, event_idx: int) -> Dict:
        """提取接管前特征"""
        start = max(0, event_idx - self.pre_window)
        segment = df.iloc[start:event_idx+1]
        
        return {
            'min_TTC': segment['TTC'].min(),
            'TTC_下降率': (segment['TTC'].iloc[-1] - segment['TTC'].iloc[0]) / max(len(segment), 1),
            'max_lane_deviation': abs(segment['车道偏离量']).max(),
            'lane_deviation_std': segment['车道偏离量'].std(),
            'max_deceleration': segment['加速度'].min(),
            'speed_at_takeover': segment['车速'].iloc[-1],
            'front_distance_min': segment['前车距离'].min(),
            'front_distance_at_takeover': segment['前车距离'].iloc[-1],
            'dominant_obj_type': segment['OD目标类型'].mode().iloc[0] if len(segment['OD目标类型'].mode()) > 0 else '无',
            'obj_changes': (segment['OD目标类型'] != segment['OD目标类型'].shift()).sum(),
        }
    
    def classify_root_cause(self, features: Dict) -> str:
        """根因分类"""
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
    
    def analyze(self, df: pd.DataFrame) -> List[TakeoverEvent]:
        """完整分析流程"""
        event_indices = self.detect_takeovers(df)
        results = []
        
        for idx in event_indices:
            features = self.extract_features(df, idx)
            root_cause = self.classify_root_cause(features)
            results.append(TakeoverEvent(
                idx=idx,
                time=df['时间戳'].iloc[idx],
                features=features,
                root_cause=root_cause
            ))
        
        return results


class SceneFeatureExtractor:
    """场景特征提取器"""
    
    def __init__(self, n_stat: int = 6, n_fft: int = 5, n_hist: int = 5):
        self.n_stat = n_stat
        self.n_fft = n_fft
        self.n_hist = n_hist
    
    def extract_signal_features(self, signal: np.ndarray) -> List[float]:
        """提取单信号特征"""
        features = [
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75)
        ]
        
        # FFT特征
        if len(signal) > 1:
            fft = np.abs(np.fft.fft(signal))
            n_comp = min(self.n_fft, len(fft) // 2)
            fft_feat = fft[1:n_comp+1] / (len(signal) + 1e-8)
            features.extend(fft_feat)
            if n_comp < self.n_fft:
                features.extend([0] * (self.n_fft - n_comp))
        else:
            features.extend([0] * self.n_fft)
        
        # 直方图特征
        hist, _ = np.histogram(signal, bins=self.n_hist)
        features.extend(hist / (len(signal) + 1e-8))
        
        return features
    
    def extract(self, scene_df: pd.DataFrame) -> np.ndarray:
        """提取场景特征向量"""
        features = []
        for col in ['车速', '加速度', '方向盘转角']:
            if col in scene_df.columns:
                features.extend(self.extract_signal_features(scene_df[col].values))
            else:
                features.extend([0] * (self.n_stat + self.n_fft + self.n_hist))
        
        features.append(len(scene_df))
        return np.array(features)
    
    def extract_batch(self, scenes: List[pd.DataFrame]) -> np.ndarray:
        """批量提取"""
        return np.array([self.extract(s) for s in scenes])


class SceneSearchEngine:
    """场景搜索引擎"""
    
    def __init__(self, extractor: SceneFeatureExtractor):
        self.extractor = extractor
        self.scenes = []
        self.feature_matrix = None
        self.scaler = StandardScaler()
    
    def index(self, scenes: List[pd.DataFrame]):
        """构建索引"""
        self.scenes = scenes
        raw = self.extractor.extract_batch(scenes)
        self.feature_matrix = self.scaler.fit_transform(raw)
        print(f'✅ 索引构建完成: {len(scenes)}个场景, 维度{self.feature_matrix.shape[1]}')
    
    def search(self, query_scene: pd.DataFrame, top_k: int = 5) -> List[Tuple[int, float, pd.DataFrame]]:
        """搜索最相似场景"""
        qf = self.extractor.extract(query_scene).reshape(1, -1)
        q_norm = self.scaler.transform(qf)[0]
        
        sims = np.dot(self.feature_matrix, q_norm) / (
            np.linalg.norm(self.feature_matrix, axis=1) * np.linalg.norm(q_norm) + 1e-8
        )
        
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx]), self.scenes[idx]) for idx in top_idx]


# ============================================================
# 第五部分：可视化报告
# ============================================================

class VisualizationReporter:
    """统一的可视化报告生成器"""
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_braking_overview(self, df: pd.DataFrame, events: List[BrakingEvent]):
        """急刹车全景图"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        
        axes[0].plot(df['时间戳'], df['车速'], 'b-', lw=0.8, alpha=0.7)
        axes[0].set_ylabel('车速 (km/h)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df['时间戳'], df['加速度'], 'g-', lw=0.8, alpha=0.7)
        axes[1].axhline(y=-4.0, color='r', linestyle='--', alpha=0.5, label='急刹阈值')
        axes[1].set_ylabel('加速度 (m/s²)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        for event in events:
            for ax in axes[:2]:
                ax.axvspan(event.start_time, event.end_time, alpha=0.2, color='red')
        
        axes[2].plot(df['时间戳'], df['方向盘转角'], 'orange', lw=0.8, alpha=0.7)
        axes[2].set_ylabel('方向盘角度 (°)')
        axes[2].set_xlabel('时间 (s)')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'急刹车场景全景图 - 共{len(events)}个事件', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'braking_overview.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✅ 急刹车全景图: {path}')
    
    def plot_takeover_analysis(self, takeover_events: List[TakeoverEvent]):
        """接管根因分析图"""
        if not takeover_events:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 饼图
        causes = [e.root_cause for e in takeover_events]
        cause_counts = Counter(causes)
        colors = plt.cm.Set2.colors[:len(cause_counts)]
        axes[0].pie(cause_counts.values(), labels=cause_counts.keys(), 
                   autopct='%1.1f%%', colors=colors)
        axes[0].set_title('接管根因分布', fontsize=13, fontweight='bold')
        
        # 柱状图
        cause_df = pd.DataFrame(cause_counts.items(), columns=['原因', '次数'])
        cause_df = cause_df.sort_values('次数', ascending=True)
        axes[1].barh(cause_df['原因'], cause_df['次数'], color=colors[:len(cause_df)])
        axes[1].set_xlabel('次数')
        axes[1].set_title('根因排序', fontsize=13, fontweight='bold')
        
        plt.suptitle(f'接管根因分析报告 - 共{len(takeover_events)}次接管', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'takeover_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✅ 接管分析图: {path}')
    
    def plot_search_results(self, query: pd.DataFrame, results: List[Tuple], 
                           feature_matrix: np.ndarray, scenes: List[pd.DataFrame],
                           query_idx: int = None):
        """相似场景搜索可视化"""
        # 检索结果对比
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()
        
        axes[0].plot(query['时间'], query['车速'], 'b-', lw=2)
        axes[0].set_title(f"查询场景 ({query['场景类型'].iloc[0]})", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        for i, (idx, score, scene) in enumerate(results):
            if i >= 5:
                break
            ax = axes[i+1]
            ax.plot(scene['时间']-scene['时间'].iloc[0], scene['车速'], 'g-', lw=2)
            ax.set_title(f'#{i+1} 相似度:{score:.3f} ({scene["场景类型"].iloc[0]})', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('相似场景检索结果', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'scene_search_results.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✅ 搜索可视化: {path}')
        
        # 特征空间
        if feature_matrix is not None:
            pca = PCA(n_components=2)
            xy = pca.fit_transform(feature_matrix)
            
            plt.figure(figsize=(10, 8))
            type_colors = {'normal': '#3498db', 'braking': '#e74c3c', 
                          'acceleration': '#2ecc71', 'turning': '#f39c12'}
            
            for t in set(s['场景类型'].iloc[0] for s in scenes):
                mask = [s['场景类型'].iloc[0]==t for s in scenes]
                plt.scatter(xy[mask,0], xy[mask,1], c=type_colors.get(t,'gray'), 
                          label=t, alpha=0.6, s=50)
            
            if query_idx is not None:
                plt.scatter(xy[query_idx,0], xy[query_idx,1], c='red', 
                          s=300, marker='*', edgecolors='black', lw=2, label='查询')
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            plt.title('场景特征空间 (PCA)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            path = os.path.join(self.output_dir, 'feature_space.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'✅ 特征空间: {path}')
    
    def generate_summary_html(self, braking_events: List, takeover_events: List,
                             scenes_count: int) -> str:
        """生成汇总HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>自动驾驶场景分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #ecf0f1; border-radius: 8px; min-width: 150px; text-align: center; }}
                .metric .value {{ font-size: 36px; font-weight: bold; color: #2980b9; }}
                .metric .label {{ color: #7f8c8d; margin-top: 5px; }}
                img {{ max-width: 100%; margin: 20px 0; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚗 自动驾驶场景数据挖掘与分析报告</h1>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="value">{len(braking_events)}</div>
                        <div class="label">急刹车事件</div>
                    </div>
                    <div class="metric">
                        <div class="value">{len(takeover_events)}</div>
                        <div class="label">接管事件</div>
                    </div>
                    <div class="metric">
                        <div class="value">{scenes_count}</div>
                        <div class="label">场景库规模</div>
                    </div>
                </div>
                
                <h2>急刹车事件分析</h2>
                <img src="braking_overview.png" alt="急刹车全景图">
                
                <h2>接管根因分析</h2>
                <img src="takeover_analysis.png" alt="接管分析">
                
                <h2>相似场景搜索</h2>
                <img src="scene_search_results.png" alt="场景搜索">
                <img src="feature_space.png" alt="特征空间">
                
                <p style="text-align: center; color: #95a5a6; margin-top: 30px;">
                    报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </body>
        </html>
        """
        
        path = os.path.join(self.output_dir, 'report.html')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'✅ HTML报告: {path}')
        return path


# ============================================================
# 第六部分：主控平台
# ============================================================

class AutonomousDrivingPlatform:
    """自动驾驶数据分析平台（整合所有功能）"""
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.data_generator = ScenarioDataGenerator()
        self.braking_detector = HardBrakingDetector()
        self.takeover_analyzer = TakeoverAnalyzer()
        self.feature_extractor = SceneFeatureExtractor()
        self.search_engine = SceneSearchEngine(self.feature_extractor)
        self.reporter = VisualizationReporter(output_dir)
        
        self.braking_events = []
        self.takeover_events = []
        self.scenes = []
    
    def run_full_pipeline(self):
        """运行完整分析流程"""
        print('='*70)
        print('🚗 自动驾驶场景数据挖掘与分析平台启动')
        print('='*70)
        
        # Step 1: 生成/加载数据
        print('\n📡 [1/5] 生成模拟行车日志...')
        df = self.data_generator.generate_log_with_events(duration_s=1800)
        print(f'   数据规模: {len(df):,} 行, {df.memory_usage(deep=True).sum()/1024:.1f} KB')
        
        # Step 2: 急刹车检测
        print('\n🔍 [2/5] 检测急刹车事件...')
        self.braking_events = self.braking_detector.detect(df)
        print(f'   检测到 {len(self.braking_events)} 个急刹车事件')
        for i, event in enumerate(self.braking_events):
            print(f'   事件{i+1}: 时间={event.start_time:.1f}s, 持续={event.duration_s:.2f}s, '
                  f'最大减速={event.max_deceleration:.1f}m/s², 速度降幅={event.speed_drop:.1f}km/h')
        
        # Step 3: 接管分析
        print('\n🔬 [3/5] 分析接管事件根因...')
        self.takeover_events = self.takeover_analyzer.analyze(df)
        print(f'   检测到 {len(self.takeover_events)} 次接管')
        cause_counts = Counter([e.root_cause for e in self.takeover_events])
        for cause, count in cause_counts.most_common():
            print(f'   {cause}: {count}次 ({count/len(self.takeover_events)*100:.1f}%)')
        
        # Step 4: 场景库与搜索引擎
        print('\n📦 [4/5] 构建场景库与搜索引擎...')
        self.scenes = self.data_generator.generate_scene_library(n_scenes=200)
        self.search_engine.index(self.scenes)
        
        # 执行搜索
        braking_scenes = [s for s in self.scenes if s['场景类型'].iloc[0] == 'braking']
        query_idx = None
        search_results = None
        
        if braking_scenes:
            query = braking_scenes[0]
            query_idx = self.scenes.index(query)
            search_results = self.search_engine.search(query)
            print(f'\n   查询场景: braking, 相似场景:')
            for i, (idx, sim, scene) in enumerate(search_results[:5]):
                print(f'   #{i+1}: 相似度={sim:.3f}, 类型={scene["场景类型"].iloc[0]}')
        
        # Step 5: 生成报告
        print('\n📊 [5/5] 生成可视化报告...')
        self.reporter.plot_braking_overview(df, self.braking_events)
        self.reporter.plot_takeover_analysis(self.takeover_events)
        
        if search_results:
            self.reporter.plot_search_results(
                query, search_results[:6], 
                self.search_engine.feature_matrix, self.scenes, query_idx
            )
        
        self.reporter.generate_summary_html(
            self.braking_events, self.takeover_events, len(self.scenes)
        )
        
        print('\n' + '='*70)
        print('✅ 分析完成！请查看 reports/ 目录')
        print('='*70)
        
        return {
            'braking_events': self.braking_events,
            'takeover_events': self.takeover_events,
            'scenes': self.scenes,
            'search_results': search_results
        }
    
    def process_large_file(self, file_path: str, chunk_size: int = 100000):
        """处理大型真实数据文件"""
        print(f'📂 处理大型文件: {file_path}')
        
        # 分块检测急刹车
        print('🔍 分块检测急刹车...')
        self.braking_events = self.braking_detector.detect_in_chunks(file_path, chunk_size)
        print(f'   共检测到 {len(self.braking_events)} 个急刹车事件')
        
        # 分块读取进行分析
        df = BigDataProcessor.load_large_csv(file_path, chunk_size=chunk_size)
        
        # 可选降采样
        if len(df) > 500000:
            print('📉 数据量较大，执行降采样...')
            df = BigDataProcessor.downsample(df, target_fs=2)
        
        self.takeover_events = self.takeover_analyzer.analyze(df)
        
        return df


# ============================================================
# 第七部分：主程序入口
# ============================================================

if __name__ == '__main__':
    start_time = time.time()
    
    platform = AutonomousDrivingPlatform(output_dir='reports')
    results = platform.run_full_pipeline()
    
    elapsed = time.time() - start_time
    print(f'\n⏱️ 总耗时: {elapsed:.1f}秒\n')
    print('📁 生成的文件:')
    for f in os.listdir('reports'):
        size = os.path.getsize(os.path.join('reports', f))
        print(f'   reports/{f} ({size/1024:.1f} KB)')
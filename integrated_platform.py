"""
自动驾驶场景数据挖掘与分析平台（ADAS传感器增强版）
支持：
  - 多传感器协议适配 (Mobileye/地平线/Continental雷达/CAN总线)
  - 根因分析 (证据链加权评分)
  - 大数据分块处理 & 文件夹并行处理
  - 单文件/文件夹/大文件/自动协议检测 四种运行模式
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
from enum import Enum
import os, sys, time, json, math, warnings, logging

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 1. 数据协议定义
# ============================================================

class SensorType(Enum):
    CAMERA_MOBILEYE = "camera_mobileye"
    CAMERA_HORIZON = "camera_horizon"
    RADAR_CONTINENTAL = "radar_continental"
    RADAR_BOSCH = "radar_bosch"
    CAN_BUS = "can_bus"
    LIDAR_HESAI = "lidar_hesai"
    UNKNOWN = "unknown"


@dataclass
class UnifiedSensorFrame:
    """统一传感器帧格式——所有异构数据转成这个"""
    timestamp: float
    sensor_type: SensorType
    objects: List[Dict]
    metadata: Dict = field(default_factory=dict)


@dataclass
class RootCauseResult:
    primary_cause: str
    confidence: float
    evidence: List[str]
    alternative_causes: List[Tuple[str, float]]


# ============================================================
# 2. 协议适配层
# ============================================================

class SensorProtocolAdapter:
    """多传感器协议适配器：异构数据 → 统一格式"""

    def detect(self, file_path: str) -> SensorType:
        name = os.path.basename(file_path).lower()
        if 'mobileye' in name or '_me_' in name: return SensorType.CAMERA_MOBILEYE
        if 'horizon' in name or '_j5_' in name or '_j6_' in name: return SensorType.CAMERA_HORIZON
        if 'radar' in name or 'ars' in name: return SensorType.RADAR_CONTINENTAL
        if 'can' in name or 'chassis' in name or 'vehicle' in name: return SensorType.CAN_BUS
        if file_path.endswith('.json'):
            with open(file_path) as f:
                data = json.load(f)
            if 'objects' in data: return SensorType.CAMERA_MOBILEYE
            if 'obstacles' in data: return SensorType.CAMERA_HORIZON
            if 'targets' in data and 'azimuth' in str(data.get('targets', [{}])[0]): return SensorType.RADAR_CONTINENTAL
        return SensorType.UNKNOWN

    def adapt(self, file_path: str) -> Union[UnifiedSensorFrame, pd.DataFrame]:
        sensor_type = self.detect(file_path)
        logger.info(f"检测到传感器类型: {sensor_type.value}")
        with open(file_path) as f:
            raw = f.read()
        if sensor_type == SensorType.CAMERA_MOBILEYE: return self._parse_mobileye(raw)
        if sensor_type == SensorType.CAMERA_HORIZON: return self._parse_horizon(raw)
        if sensor_type == SensorType.RADAR_CONTINENTAL: return self._parse_radar(raw)
        if sensor_type == SensorType.CAN_BUS: return self._parse_can(file_path)
        raise ValueError(f"不支持的传感器类型: {sensor_type}")

    def _parse_mobileye(self, raw: str) -> UnifiedSensorFrame:
        data = json.loads(raw)
        objs = [{'id': o['id'], 'type': o['type'], 'distance_x': o['x'], 'distance_y': o['y'],
                 'rel_vel_x': o.get('vx', 0), 'rel_vel_y': o.get('vy', 0),
                 'confidence': o.get('confidence', 0), 'track_age': o.get('age', 0)} for o in data.get('objects', [])]
        return UnifiedSensorFrame(data['timestamp'], SensorType.CAMERA_MOBILEYE, objs, {'source': 'mobileye'})

    def _parse_horizon(self, raw: str) -> UnifiedSensorFrame:
        data = json.loads(raw)
        objs = []
        for o in data.get('obstacles', []):
            p = o.get('polygon', {}); x, y = p.get('x', 0), p.get('y', 0)
            objs.append({'id': o['id'], 'type': o.get('type', 'unknown'), 'distance_x': x, 'distance_y': y,
                         'rel_vel_x': o.get('velocity', {}).get('x', 0), 'rel_vel_y': o.get('velocity', {}).get('y', 0),
                         'confidence': o.get('score', 0)})
        return UnifiedSensorFrame(data['header']['timestamp'], SensorType.CAMERA_HORIZON, objs, {'source': 'horizon'})

    def _parse_radar(self, raw: str) -> UnifiedSensorFrame:
        data = json.loads(raw)
        objs = []
        for o in data.get('targets', []):
            rad = math.radians(o['azimuth'])
            x = o['range'] * math.cos(rad); y = o['range'] * math.sin(rad)
            objs.append({'id': o['id'], 'type': 'radar_target', 'distance_x': round(x, 2), 'distance_y': round(y, 2),
                         'range': o['range'], 'azimuth': o['azimuth'], 'rcs': o.get('rcs', 0),
                         'rel_vel_x': o['vrel'], 'confidence': 0.9 if o.get('status') == 'stable' else 0.5,
                         'track_status': o.get('status', 'unknown')})
        return UnifiedSensorFrame(data['timestamp'], SensorType.RADAR_CONTINENTAL, objs, {'source': 'continental'})

    def _parse_can(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        mapping = {'timestamp': '时间戳', 'speed': '车速', 'brake': '刹车踏板', 'throttle': '油门踏板',
                   'steering_angle': '方向盘转角', 'gear': '档位', 'yaw_rate': '横摆角速度'}
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        logger.info(f"CAN解析: {len(df)}条, 车速 {df['车速'].min():.0f}-{df['车速'].max():.0f} km/h")
        return df


# ============================================================
# 3. 根因分析器
# ============================================================

class EvidenceBasedAnalyzer:
    def __init__(self):
        self.rules = {
            '前向碰撞风险': {'primary': [('min_TTC', '<', 2.0, 0.4), ('前车距离', '<', 10, 0.3), ('相对速度', '<', -5, 0.2)],
                             'secondary': [('摄像头目标数', '>', 1, 0.1)]},
            '车道保持异常': {'primary': [('车道偏离量', '>', 0.5, 0.4), ('方向盘转角', '>', 15, 0.2)],
                             'secondary': [('车速', '>', 40, 0.2)]},
            '传感器遮挡/退化': {'primary': [('摄像头平均置信度', '<', 0.5, 0.35), ('摄像头目标数', '<', 2, 0.25), ('雷达跟踪稳定率', '<', 0.5, 0.2)],
                               'secondary': []},
            '紧急制动触发': {'primary': [('加速度', '<', -5.0, 0.5), ('车速降幅', '>', 15, 0.3)], 'secondary': []},
            '驾驶员主动介入': {'primary': [('TTC', '>', 5.0, 0.3), ('车速', '<', 30, 0.2)],
                              'secondary': [('车道偏离量', '<', 0.3, 0.2)]},
        }

    def _check(self, v, op, t): return (v < t) if op == '<' else (v > t)

    def analyze(self, features: Dict) -> RootCauseResult:
        scores, evidence_all = {}, {}
        for cause, rule in self.rules.items():
            ev, total, mx = [], 0, 0
            for feat, op, th, w in rule['primary']:
                mx += w
                if feat in features:
                    v = features[feat]
                    if self._check(v, op, th): total += w; ev.append(f"✓ {feat}={v:.2f} {op} {th}")
                    else: ev.append(f"✗ {feat}={v:.2f} {op} {th}")
                else: ev.append(f"? {feat}")
            for feat, op, th, w in rule['secondary']:
                if feat in features and self._check(features[feat], op, th): total += w * 0.5; ev.append(f"  + {feat}")
            scores[cause] = min(1.0, total / mx if mx else 0)
            evidence_all[cause] = ev
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return RootCauseResult(ranked[0][0], round(ranked[0][1], 3), evidence_all[ranked[0][0]],
                               [(c, s) for c, s in ranked[1:3] if s > 0.2])


# ============================================================
# 4. 数据生成器
# ============================================================

class CompleteScenarioGenerator:
    def __init__(self, seed=42): np.random.seed(seed)

    def generate_complete_log(self, duration_s=600, fs=10):
        n = duration_s * fs; t = np.arange(n) / fs
        speed = np.clip(30 + 15 * np.sin(2 * np.pi * t / 300) + np.random.randn(n) * 2, 5, 70)
        acc = np.gradient(speed, 1 / fs)
        steering = 2 * np.sin(2 * np.pi * t / 45) + np.random.randn(n) * 1.5
        front_distance = np.clip(20 + 10 * np.sin(2 * np.pi * t / 90) + np.random.randn(n) * 3, 2, 50)
        rel_speed = -3 + np.random.randn(n) * 2
        ttc = np.clip(np.where(rel_speed < -0.5, front_distance / abs(rel_speed), 99), 0.5, 99)
        lane_deviation = np.clip(np.random.randn(n) * 0.2, -0.8, 0.8)
        takeover = np.zeros(n)
        cam_counts = np.zeros(n, dtype=int); cam_conf = np.zeros(n)
        radar_counts = np.zeros(n, dtype=int); radar_stable = np.zeros(n)

        self._inject_events(n, fs, acc, ttc, front_distance, lane_deviation, takeover, cam_counts, cam_conf, radar_counts, radar_stable)

        for i in range(n):
            if takeover[i] == 1:
                cam_counts[i] = np.random.randint(3, 8); cam_conf[i] = np.random.uniform(0.3, 0.7)
                radar_counts[i] = np.random.randint(4, 10); radar_stable[i] = np.random.uniform(0.3, 0.6)
            else:
                cam_counts[i] = np.random.randint(1, 5); cam_conf[i] = np.random.uniform(0.7, 0.95)
                radar_counts[i] = np.random.randint(2, 7); radar_stable[i] = np.random.uniform(0.7, 0.95)

        return pd.DataFrame({'时间戳': t, '车速': speed, '加速度': acc, '方向盘转角': steering,
                             '前车距离': front_distance, '相对速度': rel_speed, 'TTC': ttc,
                             '车道偏离量': lane_deviation, '接管标记': takeover,
                             '摄像头目标数': cam_counts, '摄像头平均置信度': cam_conf,
                             '雷达目标数': radar_counts, '雷达跟踪稳定率': radar_stable})

    def _inject_events(self, n, fs, acc, ttc, fd, ld, to, cc, cf, rc, rs):
        idx = int(200 * fs); acc[idx:idx+20] = np.linspace(-2, -7, 20)
        ttc[idx-50:idx] = np.linspace(10, 1.2, 50); fd[idx-50:idx] = np.linspace(30, 4, 50)
        to[idx+20:idx+25] = 1; cf[idx-50:idx+25] = np.linspace(0.9, 0.4, 75); rs[idx-50:idx+25] = np.linspace(0.9, 0.5, 75)
        idx = int(300 * fs); ld[idx-30:idx+10] = np.linspace(-0.1, 0.75, 40); to[idx:idx+5] = 1
        idx = int(450 * fs); length = min(30, n - idx)
        if length > 0: cf[idx:idx+length] = np.random.uniform(0.2, 0.5, length); cc[idx:idx+length] = np.random.randint(0, 2, length); rs[idx:idx+length] = np.random.uniform(0.3, 0.6, length)
        if idx + 25 < n: to[idx+25:idx+28] = 1


# ============================================================
# 5. 主控平台
# ============================================================

class ADASScenarioPlatform:
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir; os.makedirs(output_dir, exist_ok=True)
        self.data_gen = CompleteScenarioGenerator(); self.analyzer = EvidenceBasedAnalyzer()
        self.adapter = SensorProtocolAdapter()

    COL_MAP = {'时间戳': ['时间戳', 'timestamp', 'time'], '车速': ['车速', 'speed', 'velocity'],
               '加速度': ['加速度', 'acc'], '方向盘转角': ['方向盘转角', 'steering'],
               '前车距离': ['前车距离', 'front_distance'], '相对速度': ['相对速度', 'rel_speed'],
               'TTC': ['TTC', 'ttc'], '车道偏离量': ['车道偏离量', 'lane_deviation'],
               '接管标记': ['接管标记', 'takeover'], '摄像头目标数': ['摄像头目标数', 'cam_count'],
               '摄像头平均置信度': ['摄像头平均置信度', 'cam_conf'],
               '雷达目标数': ['雷达目标数', 'radar_count'], '雷达跟踪稳定率': ['雷达跟踪稳定率', 'radar_stable']}

    def _auto_rename(self, df):
        rename = {}
        for target, cands in self.COL_MAP.items():
            for c in cands:
                if c in df.columns and c != target: rename[c] = target; break
        if rename: df = df.rename(columns=rename)
        return df

    # ===== 四种运行模式 =====

    def run_demo(self):
        logger.info("演示模式：生成模拟数据...")
        df = self.data_gen.generate_complete_log()
        self._analyze_and_report(df)

    def run_csv(self, csv_path):
        logger.info(f"CSV模式：{csv_path}")
        df = pd.read_csv(csv_path); df = self._auto_rename(df)
        if '接管标记' not in df.columns: df['接管标记'] = 0
        self._analyze_and_report(df)

    def run_large_csv(self, csv_path, chunk_size=100000):
        logger.info(f"大数据模式：{csv_path} (块大小={chunk_size})")
        total, stats = 0, []
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            chunk = self._auto_rename(chunk)
            s = {'rows': len(chunk), 'cam_conf': chunk['摄像头平均置信度'].mean() if '摄像头平均置信度' in chunk.columns else 0}
            if '接管标记' in chunk.columns: s['takeover'] = int(chunk['接管标记'].sum())
            stats.append(s); total += len(chunk)
            if (i + 1) % 10 == 0: logger.info(f"  已处理 {total:,} 行")
        logger.info(f"完成: {total:,}行, 接管: {sum(s.get('takeover', 0) for s in stats)}次")
        self._summary_plot(stats, total)

    def run_folder(self, folder_path, max_workers=4):
        from concurrent.futures import ProcessPoolExecutor
        import glob
        files = glob.glob(os.path.join(folder_path, '*.csv'))
        logger.info(f"并行模式：{len(files)} 个文件, {max_workers} 进程")

        def process_one(p):
            try:
                df = pd.read_csv(p)
                df = ADASScenarioPlatform._static_rename(df)
                return {'file': os.path.basename(p), 'rows': len(df), 'takeover': int(df['接管标记'].sum()) if '接管标记' in df.columns else 0}
            except Exception as e:
                return {'file': os.path.basename(p), 'error': str(e)}

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(process_one, files))
        ok = sum(1 for r in results if 'error' not in r)
        logger.info(f"完成: {ok}/{len(files)} 个文件, 总接管: {sum(r.get('takeover', 0) for r in results)}")
        return results

    def run_auto_detect(self, file_path):
        logger.info(f"自动检测模式：{file_path}")
        sensor_type = self.adapter.detect(file_path)
        if sensor_type == SensorType.UNKNOWN: logger.error("无法识别文件类型"); return
        logger.info(f"传感器类型: {sensor_type.value}")
        if sensor_type == SensorType.CAN_BUS:
            df = self.adapter.adapt(file_path)
            if '接管标记' not in df.columns: df['接管标记'] = 0
            self._analyze_and_report(df)
        else:
            frame = self.adapter.adapt(file_path)
            logger.info(f"目标数: {len(frame.objects)}, 类型分布: {dict(Counter(o['type'] for o in frame.objects))}")

    @staticmethod
    def _static_rename(df):
        p = ADASScenarioPlatform('reports')
        return p._auto_rename(df)

    # ===== 核心分析 =====

    def _analyze_and_report(self, df):
        self._analyze_sensors(df)
        if df['接管标记'].sum() > 0: self._analyze_takeover(df)
        self._gen_reports(df)

    def _analyze_sensors(self, df):
        logger.info(f"摄像头: 目标{df['摄像头目标数'].mean():.1f} 置信度{df['摄像头平均置信度'].mean():.2f}")
        logger.info(f"雷达: 目标{df['雷达目标数'].mean():.1f} 稳定率{df['雷达跟踪稳定率'].mean():.2f}")
        anomaly = ((df['摄像头平均置信度'] < 0.5) | (df['雷达跟踪稳定率'] < 0.5)).mean() * 100
        logger.info(f"传感器异常占比: {anomaly:.1f}%")

    def _analyze_takeover(self, df):
        idxs = df[df['接管标记'] == 1].index
        events = []; i = 0
        while i < len(idxs):
            start = idxs[i]; j = i
            while j + 1 < len(idxs) and idxs[j+1] - idxs[j] == 1: j += 1
            seg = df.iloc[max(0, start-50):start+1]
            feats = {'min_TTC': seg['TTC'].min(), 'TTC': seg['TTC'].iloc[-1], '前车距离': seg['前车距离'].iloc[-1],
                     '相对速度': seg['相对速度'].iloc[-1], '车道偏离量': abs(seg['车道偏离量']).max(),
                     '加速度': seg['加速度'].min(), '车速': seg['车速'].iloc[-1],
                     '车速降幅': seg['车速'].iloc[0] - seg['车速'].iloc[-1],
                     '方向盘转角': abs(seg['方向盘转角']).max(), '摄像头目标数': seg['摄像头目标数'].mean(),
                     '摄像头平均置信度': seg['摄像头平均置信度'].mean(), '雷达跟踪稳定率': seg['雷达跟踪稳定率'].mean()}
            r = self.analyzer.analyze(feats)
            events.append((df['时间戳'].iloc[start], r)); i = j + 1
        for t, r in events:
            logger.info(f"接管 {t:.0f}s → {r.primary_cause} (置信度:{r.confidence:.2f})")

    # ===== 可视化 =====

    def _gen_reports(self, df):
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        axes[0].plot(df['时间戳'], df['车速'], 'b-', lw=0.8); axes[0].set_ylabel('车速'); axes[0].grid(True, alpha=0.3)
        axes[0].set_title('ADAS传感器性能全景分析', fontsize=14, fontweight='bold')
        axes[1].plot(df['时间戳'], df['摄像头平均置信度'], 'g-', lw=0.8); axes[1].axhline(0.6, color='orange', ls='--')
        axes[1].fill_between(df['时间戳'], 0, 1, where=(df['摄像头平均置信度'] < 0.6), color='red', alpha=0.15)
        axes[1].set_ylabel('摄像头置信度'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(df['时间戳'], df['雷达跟踪稳定率'], color='#9b59b6', lw=0.8); axes[2].axhline(0.6, color='orange', ls='--')
        axes[2].fill_between(df['时间戳'], 0, 1, where=(df['雷达跟踪稳定率'] < 0.6), color='red', alpha=0.15)
        axes[2].set_ylabel('雷达稳定率'); axes[2].grid(True, alpha=0.3)
        axes[3].plot(df['时间戳'], df['摄像头目标数'], 'g-', lw=0.8, alpha=0.7, label='摄像头')
        axes[3].plot(df['时间戳'], df['雷达目标数'], color='#9b59b6', lw=0.8, alpha=0.7, label='雷达')
        axes[3].set_ylabel('目标数'); axes[3].set_xlabel('时间(s)'); axes[3].legend(); axes[3].grid(True, alpha=0.3)
        for idx in df[df['接管标记'] == 1].index:
            for ax in axes: ax.axvline(df['时间戳'].iloc[idx], color='red', ls='--', alpha=0.4, lw=1)
        plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, 'adas_sensor_performance.png'), dpi=150); plt.close()

        # 热力图
        w = 50; nw = len(df) // w; m = np.zeros((nw, 4))
        for i in range(nw):
            s, e = i*w, (i+1)*w; seg = df.iloc[s:e]
            m[i] = [seg['车速'].mean(), seg['摄像头平均置信度'].mean(), seg['雷达跟踪稳定率'].mean(), abs(seg['摄像头目标数'].mean() - seg['雷达目标数'].mean())]
        mn = (m - m.min(0)) / (m.max(0) - m.min(0) + 1e-8)
        fig, ax = plt.subplots(figsize=(16, 6)); im = ax.imshow(mn.T, aspect='auto', cmap='RdYlGn')
        ax.set_yticks(range(4)); ax.set_yticklabels(['车速', '置信度', '稳定率', '融合差异'])
        ax.set_title('ADAS传感器指标热力图', fontsize=14, fontweight='bold')
        plt.colorbar(im, label='绿=好, 红=差'); plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sensor_fusion_heatmap.png'), dpi=150); plt.close()

        # HTML
        ap = ((df['摄像头平均置信度'] < 0.5) | (df['雷达跟踪稳定率'] < 0.5)).mean() * 100
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>ADAS报告</title>
<style>body{{font-family:'Microsoft YaHei',Arial;margin:40px;background:#f0f2f5}}
.c{{max-width:1100px;margin:auto;background:#fff;padding:30px;border-radius:12px;box-shadow:0 2px 20px rgba(0,0,0,.1)}}
h1{{color:#1a1a2e;border-bottom:4px solid #3498db;padding-bottom:15px}}
.r{{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}}
.m{{flex:1;min-width:150px;padding:20px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-radius:10px;text-align:center}}
.m .v{{font-size:32px;font-weight:bold}}.m .l{{font-size:14px;opacity:.9;margin-top:5px}}
img{{max-width:100%;margin:20px 0;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1)}}
.i{{background:#e8f4f8;border-left:4px solid #3498db;padding:15px;margin:15px 0}}
</style></head><body><div class="c">
<h1>🚗 ADAS传感器场景分析报告</h1>
<div class="r">
<div class="m"><div class="v">{df['摄像头目标数'].mean():.1f}</div><div class="l">摄像头目标数</div></div>
<div class="m"><div class="v">{df['摄像头平均置信度'].mean():.2f}</div><div class="l">摄像头置信度</div></div>
<div class="m"><div class="v">{df['雷达目标数'].mean():.1f}</div><div class="l">雷达目标数</div></div>
<div class="m"><div class="v">{ap:.1f}%</div><div class="l">传感器异常</div></div>
</div>
<div class="i">🔍 摄像头置信度 {df['摄像头平均置信度'].mean():.2f}, 异常占比 {ap:.1f}%, 接管 {int(df['接管标记'].sum())} 次</div>
<h2>传感器性能全景</h2><img src="adas_sensor_performance.png">
<h2>传感器融合热力图</h2><img src="sensor_fusion_heatmap.png">
<p style="text-align:center;color:#95a5a6">报告生成: {time.strftime('%Y-%m-%d %H:%M:%S')} | 数据: {len(df):,}帧</p>
</div></body></html>"""
        with open(os.path.join(self.output_dir, 'adas_report.html'), 'w', encoding='utf-8') as f: f.write(html)
        logger.info(f"报告已生成: {self.output_dir}/")

    def _summary_plot(self, stats, total):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        c = range(len(stats))
        axes[0].plot(c, [s.get('cam_conf', 0) for s in stats], 'g-', lw=1); axes[0].axhline(0.6, color='orange', ls='--')
        axes[0].set_title(f'大数据分析 (总{total:,}行)', fontsize=13); axes[0].grid(True, alpha=0.3)
        axes[1].plot(c, [s.get('radar_stable', 0) for s in stats], color='#9b59b6', lw=1); axes[1].axhline(0.6, color='orange', ls='--')
        axes[1].set_xlabel('数据块'); axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, 'bigdata_summary.png'), dpi=150); plt.close()


# ============================================================
# 6. 主程序入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ADAS场景分析平台')
    parser.add_argument('input', nargs='?', help='CSV文件/文件夹/传感器JSON')
    parser.add_argument('--workers', '-w', type=int, default=4, help='并行进程数')
    parser.add_argument('--chunk-size', type=int, default=100000, help='分块大小')
    parser.add_argument('--output', '-o', default='reports', help='输出目录')
    parser.add_argument('--auto-detect', '-a', action='store_true', help='自动检测传感器协议')
    parser.add_argument('--demo', action='store_true', help='运行模拟数据演示')
    args = parser.parse_args()

    platform = ADASScenarioPlatform(output_dir=args.output)

    if args.demo or not args.input:
        print('💡 用法:')
        print('   python integrated_platform.py --demo           模拟数据演示')
        print('   python integrated_platform.py 数据.csv         分析CSV')
        print('   python integrated_platform.py 文件夹/ -w 4     并行处理文件夹')
        print('   python integrated_platform.py 雷达.json -a     自动检测协议')
        print()
        platform.run_demo()
    elif args.auto_detect:
        platform.run_auto_detect(args.input)
    elif os.path.isdir(args.input):
        platform.run_folder(args.input, max_workers=args.workers)
    elif args.input.endswith('.csv'):
        size_mb = os.path.getsize(args.input) / 1024 / 1024
        if size_mb > 500:
            logger.info(f'文件较大({size_mb:.0f}MB)，自动切换大数据模式')
            platform.run_large_csv(args.input, chunk_size=args.chunk_size)
        else:
            platform.run_csv(args.input)
    else:
        platform.run_auto_detect(args.input)
"""核心逻辑单元测试"""
import unittest
import numpy as np
import pandas as pd
from integrated_platform import EvidenceBasedAnalyzer, CompleteScenarioGenerator

class TestEvidenceBasedAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = EvidenceBasedAnalyzer()
    
    def test_collision_risk(self):
        features = {'min_TTC': 1.5, '前车距离': 5, '相对速度': -8, '摄像头目标数': 3}
        result = self.analyzer.analyze(features)
        self.assertEqual(result.primary_cause, '前向碰撞风险')
        self.assertGreater(result.confidence, 0.5)
    
    def test_lane_deviation(self):
        features = {'车道偏离量': 0.8, '方向盘转角': 20, '车速': 50}
        result = self.analyzer.analyze(features)
        self.assertEqual(result.primary_cause, '车道保持异常')
    
    def test_sensor_degradation(self):
        features = {'摄像头平均置信度': 0.3, '摄像头目标数': 1, '雷达跟踪稳定率': 0.4}
        result = self.analyzer.analyze(features)
        self.assertEqual(result.primary_cause, '传感器遮挡/退化')
    
    def test_empty_features(self):
        result = self.analyzer.analyze({})
        self.assertIsNotNone(result.primary_cause)

class TestDataGenerator(unittest.TestCase):
    def test_generate_log_shape(self):
        gen = CompleteScenarioGenerator()
        df = gen.generate_complete_log(duration_s=600)
        self.assertEqual(len(df), 6000)  # 600s * 10Hz
        self.assertIn('车速', df.columns)
        self.assertIn('接管标记', df.columns)
    
    def test_generate_log_has_events(self):
        gen = CompleteScenarioGenerator()
        df = gen.generate_complete_log()
        self.assertGreater(df['接管标记'].sum(), 0)  # 至少有一个事件

if __name__ == '__main__':
    unittest.main()
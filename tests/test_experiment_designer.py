"""
实验设计模块单元测试
"""
import pytest
import numpy as np
from experiment_designer import ExperimentDesigner, ExperimentConfig, get_sample_size_summary


class TestExperimentDesigner:
    """实验设计器测试"""
    
    def test_sample_size_calculation(self):
        """测试样本量计算"""
        designer = ExperimentDesigner()
        
        n = designer.calculate_sample_size_per_group(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.8
        )
        
        # 样本量应该为正数
        assert n > 0
        # 对于 10% 基线，10% MDE，样本量应该在 10000-20000 范围
        assert 10000 <= n <= 20000
    
    def test_sample_size_with_different_mde(self):
        """测试不同 MDE 对样本量的影响"""
        designer = ExperimentDesigner()
        
        n_small_effect = designer.calculate_sample_size_per_group(
            baseline_rate=0.10,
            mde=0.05,  # 更小的效应
            alpha=0.05,
            power=0.8
        )
        
        n_large_effect = designer.calculate_sample_size_per_group(
            baseline_rate=0.10,
            mde=0.20,  # 更大的效应
            alpha=0.05,
            power=0.8
        )
        
        # 更小的效应需要更多样本
        assert n_small_effect > n_large_effect
    
    def test_sample_size_with_different_power(self):
        """测试不同统计功效对样本量的影响"""
        designer = ExperimentDesigner()
        
        n_low_power = designer.calculate_sample_size_per_group(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.7
        )
        
        n_high_power = designer.calculate_sample_size_per_group(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.9
        )
        
        # 更高的功效需要更多样本
        assert n_high_power > n_low_power
    
    def test_power_calculation(self):
        """测试统计功效计算"""
        designer = ExperimentDesigner()
        
        power = designer.calculate_power(
            n_per_group=15000,
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05
        )
        
        # 功效应该在 0 到 1 之间
        assert 0 <= power <= 1
        # 大样本量应该有较高的功效
        assert power > 0.7
    
    def test_traffic_split(self):
        """测试流量分配"""
        designer = ExperimentDesigner()
        
        variants = {
            'control': 0.5,
            'treatment_a': 0.25,
            'treatment_b': 0.25
        }
        
        allocation = designer.design_traffic_split(10000, variants)
        
        # 总样本量应该等于总流量
        assert sum(allocation.values()) == 10000
        # 对照组应该占约 50%
        assert 4900 <= allocation['control'] <= 5100
    
    def test_confidence_interval(self):
        """测试置信区间计算"""
        designer = ExperimentDesigner()
        
        lower, upper = designer.calculate_confidence_interval(
            successes=100,
            n_trials=1000,
            confidence_level=0.95
        )
        
        # 置信区间应该有效
        assert 0 <= lower <= upper <= 1
        # 点估计应该在区间内
        assert lower <= 0.10 <= upper
    
    def test_get_sample_size_summary(self):
        """测试样本量摘要"""
        config = ExperimentConfig(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.8
        )
        
        summary = get_sample_size_summary(config)
        
        # 检查返回的键
        assert 'sample_size_per_group' in summary
        assert 'total_sample_size' in summary
        assert 'achieved_power' in summary
        # 总样本量应该是每组的两倍
        assert summary['total_sample_size'] == 2 * summary['sample_size_per_group']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

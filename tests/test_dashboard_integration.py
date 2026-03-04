"""
Dashboard 集成测试
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# 导入模块进行功能测试
from experiment_designer import ExperimentDesigner, ExperimentConfig, get_sample_size_summary
from statistical_test import StatisticalTester, run_comprehensive_test
from result_visualizer import ResultVisualizer


class TestDashboardIntegration:
    """Dashboard 集成测试"""
    
    def test_full_workflow_sample_size(self):
        """测试完整工作流：样本量计算"""
        config = ExperimentConfig(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.8
        )
        
        summary = get_sample_size_summary(config)
        
        assert summary['sample_size_per_group'] > 0
        assert summary['total_sample_size'] > 0
        assert 0 <= summary['achieved_power'] <= 1
    
    def test_full_workflow_statistical_test(self):
        """测试完整工作流：统计检验"""
        # 模拟 A/B 测试数据
        control_successes = 1000
        control_total = 10000
        treatment_successes = 1200
        treatment_total = 10000
        
        result = run_comprehensive_test(
            control_successes, control_total,
            treatment_successes, treatment_total,
            alpha=0.05
        )
        
        # 验证结果结构
        assert 'frequentist' in result
        assert 'bayesian' in result
        assert 'recommendation' in result
        
        # 验证频率学派结果
        assert result['frequentist']['p_value'] < 0.05
        assert result['frequentist']['significant'] == True
        
        # 验证贝叶斯结果
        assert result['bayesian']['prob_treatment_better'] > 0.5
    
    def test_full_workflow_visualization(self):
        """测试完整工作流：可视化"""
        # 生成模拟数据
        np.random.seed(42)
        samples_treatment = np.random.beta(120, 880, 1000)
        samples_control = np.random.beta(100, 900, 1000)
        
        # 测试各种可视化
        fig_ci = ResultVisualizer.plot_confidence_intervals(
            0.10, (0.09, 0.11),
            0.12, (0.11, 0.13)
        )
        assert fig_ci is not None
        
        fig_posterior = ResultVisualizer.plot_posterior_distributions(
            samples_treatment, samples_control
        )
        assert fig_posterior is not None
        
        lift = (samples_treatment - samples_control) / samples_control
        fig_lift = ResultVisualizer.plot_lift_distribution(lift)
        assert fig_lift is not None
    
    def test_data_import_simulation(self):
        """测试数据导入模拟"""
        # 创建模拟数据集
        np.random.seed(42)
        n = 10000
        
        df = pd.DataFrame({
            'group': ['control'] * n + ['treatment'] * n,
            'converted': np.concatenate([
                np.random.binomial(1, 0.10, n),
                np.random.binomial(1, 0.11, n)
            ]),
            'date': pd.date_range('2024-01-01', periods=2*n, freq='H')[:2*n]
        })
        
        # 验证数据结构
        assert len(df) == 20000
        assert 'group' in df.columns
        assert 'converted' in df.columns
        assert df['group'].nunique() == 2
        
        # 计算汇总统计
        summary = df.groupby('group')['converted'].agg(['sum', 'count', 'mean'])
        
        assert len(summary) == 2
        assert 'sum' in summary.columns
        assert 'mean' in summary.columns
    
    def test_multiple_testing_correction_workflow(self):
        """测试多重检验校正工作流"""
        tester = StatisticalTester()
        
        # 模拟 5 次检验的 p 值
        p_values = [0.01, 0.03, 0.05, 0.08, 0.15]
        
        # Bonferroni 校正
        bonf_result = tester.bonferroni_correction(p_values, alpha=0.05)
        assert bonf_result['adjusted_alpha'] == 0.01  # 0.05 / 5
        assert len(bonf_result['adjusted_p_values']) == 5
        
        # FDR 校正
        fdr_result = tester.fdr_correction(p_values, alpha=0.05)
        assert len(fdr_result['adjusted_p_values']) == 5
        assert 'n_rejected' in fdr_result
    
    def test_sequential_analysis_workflow(self):
        """测试序贯分析工作流"""
        np.random.seed(42)
        
        # 模拟每日结果
        daily_p_values = []
        control_cumulative = 0
        treatment_cumulative = 0
        control_successes = 0
        treatment_successes = 0
        
        for day in range(14):
            daily_n = 1000
            control_new = np.random.binomial(1, 0.10, daily_n)
            treatment_new = np.random.binomial(1, 0.11, daily_n)
            
            control_successes += control_new.sum()
            treatment_successes += treatment_new.sum()
            control_cumulative += daily_n
            treatment_cumulative += daily_n
            
            tester = StatisticalTester()
            result = tester.two_proportion_z_test(
                control_successes, control_cumulative,
                treatment_successes, treatment_cumulative
            )
            
            daily_p_values.append(result.p_value)
        
        # 验证结果
        assert len(daily_p_values) == 14
        assert all(0 <= p <= 1 for p in daily_p_values)
        
        # 测试可视化
        fig = ResultVisualizer.plot_sequential_analysis(daily_p_values, alpha=0.05)
        assert fig is not None
    
    def test_experiment_config_validation(self):
        """测试实验配置验证"""
        # 有效配置
        config_valid = ExperimentConfig(
            baseline_rate=0.10,
            mde=0.10,
            alpha=0.05,
            power=0.8
        )
        assert 0 < config_valid.baseline_rate < 1
        assert 0 < config_valid.mde < 1
        assert 0 < config_valid.alpha < 1
        assert 0 < config_valid.power < 1
        
        # 边界配置
        config_edge = ExperimentConfig(
            baseline_rate=0.01,  # 很低基线
            mde=0.50,  # 很大效应
            alpha=0.01,  # 严格显著性
            power=0.95  # 高功效
        )
        summary = get_sample_size_summary(config_edge)
        assert summary['sample_size_per_group'] > 0
    
    def test_traffic_allocation(self):
        """测试流量分配"""
        designer = ExperimentDesigner()
        
        # 50/50 分配
        variants_50_50 = {'control': 0.5, 'treatment': 0.5}
        allocation = designer.design_traffic_split(10000, variants_50_50)
        assert abs(allocation['control'] - 5000) <= 1
        assert abs(allocation['treatment'] - 5000) <= 1
        
        # 多变量分配
        variants_multi = {'control': 0.4, 'treatment_a': 0.3, 'treatment_b': 0.3}
        allocation_multi = designer.design_traffic_split(10000, variants_multi)
        assert sum(allocation_multi.values()) == 10000
        assert allocation_multi['control'] == 4000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

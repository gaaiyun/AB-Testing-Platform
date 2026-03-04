"""
可视化模块单元测试
"""
import pytest
import numpy as np
from result_visualizer import ResultVisualizer
import plotly.graph_objects as go


class TestResultVisualizer:
    """结果可视化器测试"""
    
    def test_plot_confidence_intervals(self):
        """测试置信区间图"""
        fig = ResultVisualizer.plot_confidence_intervals(
            control_rate=0.10,
            control_ci=(0.09, 0.11),
            treatment_rate=0.12,
            treatment_ci=(0.11, 0.13),
            title="Test CI"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # 至少有对照组和实验组
        assert fig.layout.title.text == "Test CI"
    
    def test_plot_lift_distribution(self):
        """测试提升度分布图"""
        np.random.seed(42)
        lift_samples = np.random.normal(0.1, 0.05, 1000)
        
        fig = ResultVisualizer.plot_lift_distribution(
            lift_samples,
            title="Test Lift"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Test Lift"
    
    def test_plot_posterior_distributions(self):
        """测试后验分布图"""
        np.random.seed(42)
        samples_1 = np.random.beta(120, 880, 1000)  # 实验组
        samples_2 = np.random.beta(100, 900, 1000)  # 对照组
        
        fig = ResultVisualizer.plot_posterior_distributions(
            samples_1, samples_2,
            name_1="Treatment",
            name_2="Control",
            title="Test Posterior"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # 两个分布
        assert fig.layout.title.text == "Test Posterior"
    
    def test_plot_cumulative_results(self):
        """测试累积结果图"""
        daily_results = {
            'dates': list(range(1, 11)),
            'control_conv': [0.10 + np.random.normal(0, 0.01) for _ in range(10)],
            'treatment_conv': [0.12 + np.random.normal(0, 0.01) for _ in range(10)],
            'control_n': list(range(1000, 11000, 1000)),
            'treatment_n': list(range(1000, 11000, 1000))
        }
        
        fig = ResultVisualizer.plot_cumulative_results(
            daily_results,
            title="Test Cumulative"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # 4 条线
        assert len(fig.layout.annotations) >= 2  # 至少有 2 个子图标题
    
    def test_plot_sequential_analysis(self):
        """测试序贯分析图"""
        np.random.seed(42)
        daily_p_values = [0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
        
        fig = ResultVisualizer.plot_sequential_analysis(
            daily_p_values,
            alpha=0.05,
            title="Test Sequential"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.layout.title.text == "Test Sequential"
    
    def test_plot_sequential_analysis_with_alpha_spending(self):
        """测试带 alpha 花费的序贯分析"""
        daily_p_values = [0.10, 0.08, 0.05, 0.03, 0.02]
        alpha_spending = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        fig = ResultVisualizer.plot_sequential_analysis(
            daily_p_values,
            alpha=0.05,
            alpha_spending=alpha_spending,
            title="Test with Spending"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # p 值 + 花费边界
    
    def test_create_summary_dashboard(self):
        """测试综合仪表板"""
        test_results = {
            'frequentist': {
                'p_value': 0.03,
                'significant': True,
                'effect_size': 0.5,
                'confidence_interval': (0.01, 0.05)
            },
            'bayesian': {
                'prob_treatment_better': 0.97,
                'expected_lift': 0.15,
                'samples_1': np.random.beta(120, 880, 100),
                'samples_2': np.random.beta(100, 900, 100)
            },
            'recommendation': 'Implement treatment'
        }
        
        sample_sizes = {
            'control': 10000,
            'treatment': 10000
        }
        
        fig = ResultVisualizer.create_summary_dashboard(
            test_results,
            sample_sizes
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # 有数据
    
    def test_get_density_helper(self):
        """测试密度计算辅助函数"""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 1000)
        
        x, y = ResultVisualizer._get_density(samples, n_points=100)
        
        assert len(x) == 100
        assert len(y) == 100
        # 密度应该非负
        assert all(y >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

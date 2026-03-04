"""
结果可视化模块 - 置信区间、提升度、后验分布等
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Optional


class ResultVisualizer:
    """结果可视化器"""
    
    @staticmethod
    def plot_confidence_intervals(
        control_rate: float,
        control_ci: Tuple[float, float],
        treatment_rate: float,
        treatment_ci: Tuple[float, float],
        title: str = "Conversion Rate Confidence Intervals"
    ) -> go.Figure:
        """
        绘制置信区间图
        
        Args:
            control_rate: 对照组转化率
            control_ci: 对照组置信区间
            treatment_rate: 实验组转化率
            treatment_ci: 实验组置信区间
            title: 图表标题
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # 对照组
        fig.add_trace(go.Scatter(
            x=[control_ci[0], control_ci[1]],
            y=['Control', 'Control'],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Control',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[control_rate],
            y=['Control'],
            mode='markers',
            marker=dict(color='blue', size=12, symbol='circle'),
            name='Control Mean',
            showlegend=False
        ))
        
        # 实验组
        fig.add_trace(go.Scatter(
            x=[treatment_ci[0], treatment_ci[1]],
            y=['Treatment', 'Treatment'],
            mode='lines',
            line=dict(color='red', width=3),
            name='Treatment',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[treatment_rate],
            y=['Treatment'],
            mode='markers',
            marker=dict(color='red', size=12, symbol='circle'),
            name='Treatment Mean',
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Conversion Rate',
            yaxis_title='Group',
            height=300,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_lift_distribution(
        lift_samples: np.ndarray,
        title: str = "Lift Distribution"
    ) -> go.Figure:
        """
        绘制提升度分布图
        
        Args:
            lift_samples: 提升度样本
            title: 图表标题
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=lift_samples * 100,  # 转换为百分比
            nbinsx=50,
            name='Lift',
            marker_color='green',
            opacity=0.7
        ))
        
        # 添加 0 线
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="No Lift")
        
        # 添加期望值线
        mean_lift = np.mean(lift_samples) * 100
        fig.add_vline(x=mean_lift, line_dash="solid", line_color="blue",
                     annotation_text=f"Mean: {mean_lift:.2f}%")
        
        fig.update_layout(
            title=title,
            xaxis_title='Lift (%)',
            yaxis_title='Frequency',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_posterior_distributions(
        samples_1: np.ndarray,
        samples_2: np.ndarray,
        name_1: str = "Treatment",
        name_2: str = "Control",
        title: str = "Posterior Distributions"
    ) -> go.Figure:
        """
        绘制后验分布图
        
        Args:
            samples_1: 组 1 后验样本
            samples_2: 组 2 后验样本
            name_1: 组 1 名称
            name_2: 组 2 名称
            title: 图表标题
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # 计算密度
        def get_density(samples, n_points=200):
            hist, bin_edges = np.histogram(samples, bins=n_points, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return bin_centers, hist
        
        x1, y1 = get_density(samples_1)
        x2, y2 = get_density(samples_2)
        
        fig.add_trace(go.Scatter(
            x=x1, y=y1,
            mode='lines',
            name=name_1,
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=x2, y=y2,
            mode='lines',
            name=name_2,
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Conversion Rate',
            yaxis_title='Density',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_cumulative_results(
        daily_results: Dict,
        title: str = "Cumulative Results Over Time"
    ) -> go.Figure:
        """
        绘制累积结果随时间变化图
        
        Args:
            daily_results: 每日结果字典
                {
                    'dates': [...],
                    'control_conv': [...],
                    'treatment_conv': [...],
                    'control_n': [...],
                    'treatment_n': [...]
                }
            title: 图表标题
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Conversion Rate', 'Sample Size'),
                           vertical_spacing=0.15)
        
        dates = daily_results.get('dates', list(range(len(daily_results['control_conv']))))
        
        # 转化率
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_results['control_conv'],
            mode='lines+markers',
            name='Control',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_results['treatment_conv'],
            mode='lines+markers',
            name='Treatment',
            line=dict(color='red')
        ), row=1, col=1)
        
        # 样本量
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_results['control_n'],
            mode='lines',
            name='Control N',
            line=dict(color='blue', dash='dash'),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_results['treatment_n'],
            mode='lines',
            name='Treatment N',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_sequential_analysis(
        daily_p_values: list,
        alpha: float = 0.05,
        alpha_spending: Optional[list] = None,
        title: str = "Sequential Analysis"
    ) -> go.Figure:
        """
        绘制序贯分析图
        
        Args:
            daily_p_values: 每日 p 值列表
            alpha: 显著性水平
            alpha_spending: alpha 花费函数值 (可选)
            title: 图表标题
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        n_days = len(daily_p_values)
        days = list(range(1, n_days + 1))
        
        # p 值
        fig.add_trace(go.Scatter(
            x=days,
            y=daily_p_values,
            mode='lines+markers',
            name='P-value',
            line=dict(color='blue', width=2)
        ))
        
        # 显著性线
        fig.add_hline(y=alpha, line_dash="dash", line_color="red",
                     annotation_text=f"Alpha = {alpha}")
        
        # Alpha 花费边界 (如果有)
        if alpha_spending:
            fig.add_trace(go.Scatter(
                x=days,
                y=alpha_spending,
                mode='lines',
                name='Alpha Spending Boundary',
                line=dict(color='orange', dash='dot')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Day',
            yaxis_title='P-value',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_summary_dashboard(
        test_results: Dict,
        sample_sizes: Dict
    ) -> go.Figure:
        """
        创建综合摘要仪表板
        
        Args:
            test_results: 检验结果字典
            sample_sizes: 样本量信息
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Conversion Rate Comparison',
                'Lift Distribution',
                'Posterior Distributions',
                'Key Metrics'
            ),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 从结果中提取数据 (假设有 samples)
        if 'bayesian' in test_results and 'samples_1' in test_results['bayesian']:
            samples_1 = test_results['bayesian']['samples_1']
            samples_2 = test_results['bayesian']['samples_2']
            
            # 后验分布
            x1, y1 = ResultVisualizer._get_density(samples_1)
            x2, y2 = ResultVisualizer._get_density(samples_2)
            
            fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='Treatment',
                                    line=dict(color='red')), row=2, col=1)
            fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='Control',
                                    line=dict(color='blue')), row=2, col=1)
            
            # 提升度分布
            lift = (samples_1 - samples_2) / samples_2
            fig.add_trace(go.Histogram(x=lift * 100, nbinsx=50, marker_color='green'),
                         row=1, col=2)
        
        # 关键指标表格
        metrics = [
            ['Metric', 'Value'],
            ['P-value', f"{test_results.get('frequentist', {}).get('p_value', 'N/A'):.4f}"],
            ['Significant', str(test_results.get('frequentist', {}).get('significant', 'N/A'))],
            ['Prob. Better', f"{test_results.get('bayesian', {}).get('prob_treatment_better', 0):.2%}"],
            ['Expected Lift', f"{test_results.get('bayesian', {}).get('expected_lift', 0):.2%}"],
            ['Recommendation', test_results.get('recommendation', 'N/A')]
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=metrics[0], fill_color='paleturquoise', align='left'),
            cells=dict(values=[[row[0] for row in metrics[1:]], [row[1] for row in metrics[1:]]],
                      fill_color='lavender', align='left')
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="A/B Test Analysis Dashboard"
        )
        
        return fig
    
    @staticmethod
    def _get_density(samples, n_points=200):
        """辅助函数：计算密度"""
        hist, bin_edges = np.histogram(samples, bins=n_points, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

"""
实验设计模块 - 样本量计算和分流设计
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """实验配置参数"""
    baseline_rate: float  # 基线转化率
    mde: float  # 最小可检测效应
    alpha: float = 0.05  # 显著性水平
    power: float = 0.8  # 统计功效
    allocation_ratio: float = 0.5  # 实验组/对照组分配比例


class ExperimentDesigner:
    """实验设计器"""
    
    @staticmethod
    def calculate_sample_size_per_group(
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        allocation_ratio: float = 0.5
    ) -> int:
        """
        计算每组所需样本量
        
        Args:
            baseline_rate: 基线转化率
            mde: 最小可检测效应 (relative)
            alpha: 显著性水平
            power: 统计功效
            allocation_ratio: 实验组分配比例
            
        Returns:
            每组所需样本量
        """
        # 计算效应量
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        
        # 合并方差
        p_pool = (p1 + p2) / 2
        
        # Z 值
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # 样本量公式
        n = ((z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + 
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / \
            ((p2 - p1) ** 2)
        
        # 调整分配比例
        if allocation_ratio != 0.5:
            n = n * (1 + allocation_ratio) / (4 * allocation_ratio)
        
        return int(np.ceil(n))
    
    @staticmethod
    def calculate_sample_size_continuous(
        effect_size: float,
        std_dev: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        计算连续变量的样本量 (基于 Cohen's d)
        
        Args:
            effect_size: 效应量 (均值差)
            std_dev: 标准差
            alpha: 显著性水平
            power: 统计功效
            
        Returns:
            每组所需样本量
        """
        d = effect_size / std_dev  # Cohen's d
        n = 2 * ((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / d) ** 2
        return int(np.ceil(n))
    
    @staticmethod
    def calculate_power(
        n_per_group: int,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05
    ) -> float:
        """
        计算统计功效
        
        Args:
            n_per_group: 每组样本量
            baseline_rate: 基线转化率
            mde: 最小可检测效应
            alpha: 显著性水平
            
        Returns:
            统计功效
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        p_pool = (p1 + p2) / 2
        
        se_pool = np.sqrt(2 * p_pool * (1 - p_pool) / n_per_group)
        se_unpooled = np.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = (abs(p2 - p1) - z_alpha * se_pool) / se_unpooled
        
        power = stats.norm.cdf(z_beta)
        return power
    
    @staticmethod
    def design_traffic_split(
        total_traffic: int,
        variants: Dict[str, float]
    ) -> Dict[str, int]:
        """
        设计流量分配
        
        Args:
            total_traffic: 总流量
            variants: 变体及其分配比例 {'control': 0.5, 'treatment': 0.5}
            
        Returns:
            各变体的样本量
        """
        allocation = {}
        remaining = total_traffic
        
        for i, (name, ratio) in enumerate(variants.items()):
            if i == len(variants) - 1:
                allocation[name] = remaining
            else:
                n = int(total_traffic * ratio)
                allocation[name] = n
                remaining -= n
        
        return allocation
    
    @staticmethod
    def calculate_confidence_interval(
        successes: int,
        n_trials: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        计算比率的置信区间 (Wilson score interval)
        
        Args:
            successes: 成功次数
            n_trials: 总试验次数
            confidence_level: 置信水平
            
        Returns:
            (lower_bound, upper_bound)
        """
        p = successes / n_trials
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        denominator = 1 + z**2 / n_trials
        center = (p + z**2 / (2 * n_trials)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_trials + z**2 / (4 * n_trials**2)) / denominator
        
        return max(0, center - margin), min(1, center + margin)


def get_sample_size_summary(config: ExperimentConfig) -> Dict:
    """
    获取样本量计算摘要
    
    Returns:
        包含详细信息的字典
    """
    designer = ExperimentDesigner()
    
    n_per_group = designer.calculate_sample_size_per_group(
        config.baseline_rate,
        config.mde,
        config.alpha,
        config.power,
        config.allocation_ratio
    )
    
    power = designer.calculate_power(
        n_per_group,
        config.baseline_rate,
        config.mde,
        config.alpha
    )
    
    return {
        'sample_size_per_group': n_per_group,
        'total_sample_size': int(n_per_group * 2),
        'achieved_power': power,
        'baseline_rate': config.baseline_rate,
        'mde': config.mde,
        'alpha': config.alpha,
        'allocation_ratio': config.allocation_ratio
    }

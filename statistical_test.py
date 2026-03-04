"""
统计检验模块 - T 检验、卡方检验、贝叶斯方法
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TestResult:
    """检验结果"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    additional_info: Optional[Dict] = None


class StatisticalTester:
    """统计检验器"""
    
    @staticmethod
    def two_proportion_z_test(
        successes_1: int, n_1: int,
        successes_2: int, n_2: int,
        alpha: float = 0.05
    ) -> TestResult:
        """
        两比例 Z 检验
        
        Args:
            successes_1: 组 1 成功次数
            n_1: 组 1 总次数
            successes_2: 组 2 成功次数
            n_2: 组 2 总次数
            alpha: 显著性水平
            
        Returns:
            TestResult 对象
        """
        p1 = successes_1 / n_1
        p2 = successes_2 / n_2
        
        # 合并比例
        p_pool = (successes_1 + successes_2) / (n_1 + n_2)
        
        # 标准误
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_1 + 1/n_2))
        
        # Z 统计量
        z = (p1 - p2) / se
        
        # p 值 (双尾)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # 置信区间
        se_diff = np.sqrt(p1 * (1 - p1) / n_1 + p2 * (1 - p2) / n_2)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = (p1 - p2 - z_crit * se_diff, p1 - p2 + z_crit * se_diff)
        
        # 效应量 (Cohen's h)
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        effect_size = phi1 - phi2
        
        return TestResult(
            test_name='Two-Proportion Z-Test',
            statistic=z,
            p_value=p_value,
            significant=p_value < alpha,
            confidence_interval=ci,
            effect_size=effect_size,
            additional_info={
                'p1': p1,
                'p2': p2,
                'difference': p1 - p2,
                'relative_lift': (p1 - p2) / p2 if p2 > 0 else None
            }
        )
    
    @staticmethod
    def chi_square_test(
        observed: np.ndarray,
        expected: Optional[np.ndarray] = None,
        alpha: float = 0.05
    ) -> TestResult:
        """
        卡方检验
        
        Args:
            observed: 观测频数
            expected: 期望频数 (可选)
            alpha: 显著性水平
            
        Returns:
            TestResult 对象
        """
        if expected is None:
            # 拟合优度检验
            expected = np.full_like(observed, np.mean(observed))
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        # Cramer's V (效应量)
        n = np.sum(observed)
        k = len(observed)
        effect_size = np.sqrt(chi2 / n) if n > 0 else 0
        
        return TestResult(
            test_name='Chi-Square Test',
            statistic=chi2,
            p_value=p_value,
            significant=p_value < alpha,
            effect_size=effect_size
        )
    
    @staticmethod
    def t_test_independent(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        alpha: float = 0.05,
        equal_var: bool = True
    ) -> TestResult:
        """
        独立样本 T 检验
        
        Args:
            sample_1: 样本 1
            sample_2: 样本 2
            alpha: 显著性水平
            equal_var: 是否假设方差齐性
            
        Returns:
            TestResult 对象
        """
        t_stat, p_value = stats.ttest_ind(sample_1, sample_2, equal_var=equal_var)
        
        # 置信区间
        mean_diff = np.mean(sample_1) - np.mean(sample_2)
        n1, n2 = len(sample_1), len(sample_2)
        
        if equal_var:
            # 合并方差
            var1, var2 = np.var(sample_1, ddof=1), np.var(sample_2, ddof=1)
            sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            se = sp * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(np.var(sample_1, ddof=1)/n1 + np.var(sample_2, ddof=1)/n2)
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        
        # 效应量 (Cohen's d)
        pooled_std = np.sqrt((np.var(sample_1, ddof=1) + np.var(sample_2, ddof=1)) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return TestResult(
            test_name='Independent T-Test',
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < alpha,
            confidence_interval=ci,
            effect_size=effect_size,
            additional_info={
                'mean_1': np.mean(sample_1),
                'mean_2': np.mean(sample_2),
                'df': df
            }
        )
    
    @staticmethod
    def bayesian_ab_test(
        successes_1: int, n_1: int,
        successes_2: int, n_2: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 10000
    ) -> Dict:
        """
        贝叶斯 A/B 检验
        
        Args:
            successes_1: 组 1 成功次数
            n_1: 组 1 总次数
            successes_2: 组 2 成功次数
            n_2: 组 2 总次数
            prior_alpha: Beta 分布先验 alpha
            prior_beta: Beta 分布先验 beta
            n_samples: MCMC 样本数
            
        Returns:
            贝叶斯分析结果字典
        """
        # 后验分布参数
        alpha_1 = successes_1 + prior_alpha
        beta_1 = n_1 - successes_1 + prior_beta
        alpha_2 = successes_2 + prior_alpha
        beta_2 = n_2 - successes_2 + prior_beta
        
        # 从后验分布采样
        samples_1 = np.random.beta(alpha_1, beta_1, n_samples)
        samples_2 = np.random.beta(alpha_2, beta_2, n_samples)
        
        # 计算概率
        prob_better = np.mean(samples_1 > samples_2)
        diff_samples = samples_1 - samples_2
        
        # 可信区间
        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)
        
        # 期望提升
        expected_lift = np.mean(diff_samples) / (np.mean(samples_2) if np.mean(samples_2) > 0 else 1)
        
        return {
            'prob_treatment_better': prob_better,
            'expected_lift': expected_lift,
            'credible_interval': (ci_lower, ci_upper),
            'posterior_mean_1': np.mean(samples_1),
            'posterior_mean_2': np.mean(samples_2),
            'rope_prob': np.mean((diff_samples > -0.01) & (diff_samples < 0.01)),  # ROPE 概率
            'samples_1': samples_1,
            'samples_2': samples_2
        }
    
    @staticmethod
    def bonferroni_correction(p_values: list, alpha: float = 0.05) -> Dict:
        """
        Bonferroni 校正
        
        Args:
            p_values: p 值列表
            alpha: 原始显著性水平
            
        Returns:
            校正结果
        """
        m = len(p_values)
        adjusted_alpha = alpha / m
        adjusted_p_values = [min(p * m, 1.0) for p in p_values]
        
        return {
            'method': 'Bonferroni',
            'original_alpha': alpha,
            'adjusted_alpha': adjusted_alpha,
            'adjusted_p_values': adjusted_p_values,
            'significant': [p < adjusted_alpha for p in adjusted_p_values],
            'n_tests': m
        }
    
    @staticmethod
    def fdr_correction(p_values: list, alpha: float = 0.05) -> Dict:
        """
        FDR (False Discovery Rate) 校正 - Benjamini-Hochberg
        
        Args:
            p_values: p 值列表
            alpha: 显著性水平
            
        Returns:
            校正结果
        """
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        # 计算临界值
        ranks = np.arange(1, m + 1)
        critical_values = (ranks / m) * alpha
        
        # 找到最大的 k 使得 p(k) <= k/m * alpha
        significant = sorted_p <= critical_values
        if np.any(significant):
            k = np.max(np.where(significant))
            threshold = sorted_p[k]
        else:
            threshold = 0
        
        # 计算 adjusted p-values
        adjusted_p = np.zeros(m)
        for i in range(m - 1, -1, -1):
            adjusted_p[i] = min(1.0, sorted_p[i] * m / ranks[i])
            if i > 0:
                adjusted_p[i] = min(adjusted_p[i], adjusted_p[i - 1] * (ranks[i] / ranks[i-1]))
        
        # 恢复原始顺序
        adjusted_p_original = np.zeros(m)
        adjusted_p_original[sorted_indices] = adjusted_p
        
        return {
            'method': 'Benjamini-Hochberg FDR',
            'original_alpha': alpha,
            'adjusted_p_values': adjusted_p_original.tolist(),
            'significant': adjusted_p_original < alpha,
            'threshold': threshold,
            'n_rejected': np.sum(adjusted_p_original < alpha)
        }


def run_comprehensive_test(
    control_successes: int, control_total: int,
    treatment_successes: int, treatment_total: int,
    alpha: float = 0.05
) -> Dict:
    """
    运行综合检验
    
    Returns:
        包含所有检验结果的字典
    """
    tester = StatisticalTester()
    
    # 频率学派检验
    z_result = tester.two_proportion_z_test(
        control_successes, control_total,
        treatment_successes, treatment_total,
        alpha
    )
    
    # 贝叶斯检验 (treatment 是组 1, control 是组 2)
    bayesian_result = tester.bayesian_ab_test(
        treatment_successes, treatment_total,
        control_successes, control_total
    )
    
    return {
        'frequentist': {
            'test_name': z_result.test_name,
            'statistic': z_result.statistic,
            'p_value': z_result.p_value,
            'significant': z_result.significant,
            'confidence_interval': z_result.confidence_interval,
            'effect_size': z_result.effect_size,
            'additional_info': z_result.additional_info
        },
        'bayesian': {
            'prob_treatment_better': bayesian_result['prob_treatment_better'],
            'expected_lift': bayesian_result['expected_lift'],
            'credible_interval': bayesian_result['credible_interval'],
            'posterior_mean_control': bayesian_result['posterior_mean_2'],
            'posterior_mean_treatment': bayesian_result['posterior_mean_1']
        },
        'recommendation': 'Implement treatment' if (z_result.significant and bayesian_result['prob_treatment_better'] > 0.95) else 'Continue testing' if (bayesian_result['prob_treatment_better'] > 0.8) else 'Keep control'
    }

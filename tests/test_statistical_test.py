"""
统计检验模块单元测试
"""
import pytest
import numpy as np
from statistical_test import StatisticalTester, run_comprehensive_test, TestResult


class TestStatisticalTester:
    """统计检验器测试"""
    
    def test_two_proportion_z_test_significant(self):
        """测试两比例 Z 检验 - 显著情况"""
        tester = StatisticalTester()
        
        result = tester.two_proportion_z_test(
            successes_1=1200, n_1=10000,  # 12% 转化率
            successes_2=1000, n_2=10000,  # 10% 转化率
            alpha=0.05
        )
        
        assert isinstance(result, TestResult)
        assert result.test_name == 'Two-Proportion Z-Test'
        assert result.p_value < 0.05  # 应该显著
        assert result.significant == True
        assert result.confidence_interval is not None
    
    def test_two_proportion_z_test_not_significant(self):
        """测试两比例 Z 检验 - 不显著情况"""
        tester = StatisticalTester()
        
        result = tester.two_proportion_z_test(
            successes_1=1010, n_1=10000,  # 10.1% 转化率
            successes_2=1000, n_2=10000,  # 10% 转化率
            alpha=0.05
        )
        
        # 小差异应该不显著
        assert result.p_value > 0.05
        assert result.significant == False
    
    def test_chi_square_test(self):
        """测试卡方检验"""
        tester = StatisticalTester()
        
        observed = np.array([50, 50, 50, 50])
        expected = np.array([40, 60, 40, 60])
        
        result = tester.chi_square_test(observed, expected)
        
        assert result.test_name == 'Chi-Square Test'
        assert result.statistic > 0
        assert 0 <= result.p_value <= 1
    
    def test_t_test_independent(self):
        """测试独立样本 T 检验"""
        tester = StatisticalTester()
        
        np.random.seed(42)
        sample_1 = np.random.normal(100, 15, 100)
        sample_2 = np.random.normal(105, 15, 100)
        
        result = tester.t_test_independent(sample_1, sample_2)
        
        assert result.test_name == 'Independent T-Test'
        assert result.confidence_interval is not None
        assert result.effect_size is not None
    
    def test_bayesian_ab_test(self):
        """测试贝叶斯 A/B 检验"""
        tester = StatisticalTester()
        
        result = tester.bayesian_ab_test(
            successes_1=1200, n_1=10000,
            successes_2=1000, n_2=10000,
            n_samples=1000
        )
        
        assert 'prob_treatment_better' in result
        assert 'expected_lift' in result
        assert 'credible_interval' in result
        assert 0 <= result['prob_treatment_better'] <= 1
        assert isinstance(result['credible_interval'], tuple)
    
    def test_bayesian_ab_test_with_clear_winner(self):
        """测试贝叶斯检验 - 明显优胜"""
        tester = StatisticalTester()
        
        result = tester.bayesian_ab_test(
            successes_1=1500, n_1=10000,  # 15% 转化率
            successes_2=1000, n_2=10000,  # 10% 转化率
            n_samples=5000
        )
        
        # 实验组应该有很大概率更优
        assert result['prob_treatment_better'] > 0.95
    
    def test_bonferroni_correction(self):
        """测试 Bonferroni 校正"""
        tester = StatisticalTester()
        
        p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
        
        result = tester.bonferroni_correction(p_values, alpha=0.05)
        
        assert result['method'] == 'Bonferroni'
        assert result['n_tests'] == 5
        assert result['adjusted_alpha'] == 0.01  # 0.05 / 5
        assert len(result['adjusted_p_values']) == 5
        # 调整后的 p 值应该更大或等于原值
        for orig, adj in zip(p_values, result['adjusted_p_values']):
            assert adj >= orig

    def test_bonferroni_uses_adjusted_p_or_adjusted_alpha_once(self):
        """调整后的 p 值应和原 alpha 比较，不能再除一次检验数。"""
        result = StatisticalTester.bonferroni_correction([0.009, 0.02], alpha=0.05)
        assert result['adjusted_p_values'] == pytest.approx([0.018, 0.04])
        assert result['significant'] == [True, True]
    
    def test_fdr_correction(self):
        """测试 FDR 校正"""
        tester = StatisticalTester()
        
        p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
        
        result = tester.fdr_correction(p_values, alpha=0.05)
        
        assert 'Benjamini-Hochberg' in result['method']
        assert len(result['adjusted_p_values']) == 5
        assert 'n_rejected' in result

    def test_fdr_matches_benjamini_hochberg_step_up_values(self):
        result = StatisticalTester.fdr_correction([0.01, 0.02, 0.04, 0.20], alpha=0.05)
        assert result['adjusted_p_values'] == pytest.approx(
            [0.04, 0.04, 0.0533333333, 0.20]
        )
        assert result['significant'].tolist() == [True, True, False, False]
        assert result['n_rejected'] == 2
    
    def test_run_comprehensive_test(self):
        """测试综合检验"""
        result = run_comprehensive_test(
            control_successes=1000, control_total=10000,
            treatment_successes=1200, treatment_total=10000,
            alpha=0.05
        )
        
        assert 'frequentist' in result
        assert 'bayesian' in result
        assert 'recommendation' in result
        assert result['frequentist']['significant'] == True
        # 贝叶斯概率应该较高（实验组更好）
        assert result['bayesian']['prob_treatment_better'] > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

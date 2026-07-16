"""cuped.py 测试 — 验证方差缩减与无偏性。"""
import numpy as np
import pytest

from cuped import (
    CupedResult,
    cuped_adjust,
    cuped_two_sample,
    estimate_sample_size_savings,
    estimate_theta,
)


def _make_data(seed=0, n=2000, rho_target=0.7, true_effect=0.3):
    rng = np.random.default_rng(seed)
    x_c = rng.normal(10, 3, n)
    x_t = rng.normal(10, 3, n)
    y_c = x_c * rho_target + rng.normal(0, 3 * np.sqrt(1 - rho_target ** 2), n)
    y_t = x_t * rho_target + rng.normal(true_effect, 3 * np.sqrt(1 - rho_target ** 2), n)
    return x_c, y_c, x_t, y_t


def test_theta_estimation_recovers_covariance():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = 2.0 * x + rng.normal(0, 0.1, 5000)
    assert abs(estimate_theta(x, y) - 2.0) < 0.05


def test_theta_zero_when_no_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    assert abs(estimate_theta(x, y)) < 0.1


def test_cuped_unbiased_estimate_of_diff():
    """CUPED 调整后的均值差应该（在大样本下）仍然无偏。"""
    diffs = []
    for seed in range(15):
        x_c, y_c, x_t, y_t = _make_data(seed=seed, n=3000, true_effect=0.4)
        r = cuped_two_sample(x_c, y_c, x_t, y_t)
        diffs.append(r.adjusted_diff)
    # 平均应当接近真实效应 0.4
    assert abs(np.mean(diffs) - 0.4) < 0.05


def test_cuped_variance_reduction_when_correlated():
    x_c, y_c, x_t, y_t = _make_data(seed=42, n=5000, rho_target=0.7)
    r = cuped_two_sample(x_c, y_c, x_t, y_t)
    assert r.variance_reduction > 0.3, f"ρ≈0.7 应至少减少 30% 方差，实际 {r.variance_reduction:.2%}"
    assert r.correlation > 0.5


def test_cuped_no_reduction_when_uncorrelated():
    rng = np.random.default_rng(0)
    n = 3000
    x_c = rng.normal(0, 1, n)
    y_c = rng.normal(0, 1, n)
    x_t = rng.normal(0, 1, n)
    y_t = rng.normal(0.2, 1, n)
    r = cuped_two_sample(x_c, y_c, x_t, y_t)
    assert r.variance_reduction < 0.05
    assert isinstance(r, CupedResult)


def test_cuped_length_mismatch_raises():
    with pytest.raises(ValueError):
        cuped_two_sample(np.zeros(10), np.zeros(11), np.zeros(10), np.zeros(10))


def test_sample_size_savings_returns_rho_squared():
    assert estimate_sample_size_savings(0.5) == pytest.approx(0.25)
    assert estimate_sample_size_savings(0.0) == 0.0
    assert estimate_sample_size_savings(-0.7) == pytest.approx(0.49)
    assert estimate_sample_size_savings(1.0) == pytest.approx(0.99)  # 被 clamp


def test_cuped_constant_groups_with_different_means_are_significant():
    result = cuped_two_sample(
        np.zeros(4), np.ones(4),
        np.zeros(4), np.full(4, 2.0),
    )
    assert result.adjusted_diff == pytest.approx(1.0)
    assert np.isinf(result.t_stat)
    assert result.p_value == 0.0

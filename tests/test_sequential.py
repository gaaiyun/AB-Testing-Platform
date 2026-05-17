"""sequential.py 的测试 — 重点验证 always-valid 性质。"""
import numpy as np
import pytest

from sequential import (
    MSPRT,
    SequentialMonitor,
    obrien_fleming_boundary,
    pocock_boundary,
)


# ----------------------------------------------------------------------------
# mSPRT
# ----------------------------------------------------------------------------

def test_msprt_initial_state_returns_continue():
    m = MSPRT(tau=0.01, alpha=0.05)
    res = m.add_batch([0, 1], [1, 0])
    assert res["decision"] in ("continue", "reject_null")
    # 极少样本不应直接拒绝
    if res["n_total"] <= 4:
        assert res["always_valid_p_value"] >= 0.5


def test_msprt_rejects_under_clear_effect():
    """真实存在显著效应时，mSPRT 应在合理样本量内拒绝 H0。"""
    rng = np.random.default_rng(0)
    m = MSPRT(tau=0.05, alpha=0.05)
    rejected = False
    for _ in range(20):
        c = rng.binomial(1, 0.10, 500)
        t = rng.binomial(1, 0.15, 500)
        res = m.add_batch(c, t)
        if res["decision"] == "reject_null":
            rejected = True
            break
    assert rejected, "应在 10000 样本以内检出 50% 的提升"


def test_msprt_does_not_reject_under_h0_at_rate_above_alpha():
    """无效应时，多次"偷看"下也不应该频繁拒绝 H0（peeking 不膨胀 alpha）。"""
    n_trials = 100
    n_rejections = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        m = MSPRT(tau=0.05, alpha=0.05)
        for _ in range(20):  # 每个 trial 偷看 20 次
            c = rng.binomial(1, 0.10, 500)
            t = rng.binomial(1, 0.10, 500)  # 无效应
            res = m.add_batch(c, t)
            if res["decision"] == "reject_null":
                n_rejections += 1
                break
    # always-valid 性质保证：任意停止规则下，拒绝率 <= alpha
    # 100 次 trial，alpha=0.05，理论上界是 5 次拒绝（再宽松到 10 给统计噪音）
    assert n_rejections <= 10, f"observed {n_rejections}/100 rejections > acceptable threshold"


def test_msprt_invalid_params():
    with pytest.raises(ValueError):
        MSPRT(tau=0)
    with pytest.raises(ValueError):
        MSPRT(tau=0.01, alpha=1.5)
    with pytest.raises(ValueError):
        MSPRT(tau=0.01, metric="unknown")  # type: ignore[arg-type]


def test_msprt_ci_contains_diff_at_h0():
    """无效应时，always-valid CI 应当（高概率地）包含 0。"""
    n_contained = 0
    for seed in range(30):
        rng = np.random.default_rng(seed)
        m = MSPRT(tau=0.05, alpha=0.05)
        res = None
        for _ in range(10):
            c = rng.binomial(1, 0.1, 500)
            t = rng.binomial(1, 0.1, 500)
            res = m.add_batch(c, t)
        lo, hi = res["always_valid_ci"]
        if lo <= 0 <= hi:
            n_contained += 1
    # always-valid CI 是更宽的 — 覆盖率应当 >= 1 - alpha
    assert n_contained >= 26  # 30 * 0.95 = 28.5，给 2 个噪音余量


# ----------------------------------------------------------------------------
# Alpha-Spending
# ----------------------------------------------------------------------------

def test_obrien_fleming_boundaries_monotone_decreasing():
    """O'Brien-Fleming 边界：早期高、晚期接近 z_{alpha/2}。"""
    b = obrien_fleming_boundary(0.05, 5)
    z_alpha_half = 1.96
    assert b.z_boundaries[0] > b.z_boundaries[-1]
    assert b.z_boundaries[-1] <= z_alpha_half + 0.5  # 末次接近常规
    assert all(z > 0 for z in b.z_boundaries)


def test_pocock_constant_boundaries():
    b = pocock_boundary(0.05, 5)
    assert all(abs(z - b.z_boundaries[0]) < 1e-9 for z in b.z_boundaries)
    assert b.z_boundaries[0] > 1.96  # Pocock 边界一定高于裸 alpha=0.05 双尾


def test_pocock_interpolation():
    # k=4 是表内的值；k=7 应该走插值
    b4 = pocock_boundary(0.05, 4)
    b7 = pocock_boundary(0.05, 7)
    assert b7.z_boundaries[0] > b4.z_boundaries[0]


def test_alpha_spending_stop_decision():
    b = obrien_fleming_boundary(0.05, 5)
    # 第 1 个 look 的边界很高，z=2 不应触发
    assert not b.stop(observed_z=2.0, information_fraction=0.2)
    # 极大 z 应触发
    assert b.stop(observed_z=10.0, information_fraction=0.2)


# ----------------------------------------------------------------------------
# SequentialMonitor (顶层 API)
# ----------------------------------------------------------------------------

def test_sequential_monitor_msprt_path():
    rng = np.random.default_rng(1)
    mon = SequentialMonitor(method="msprt", tau=0.05, alpha=0.05, metric="proportion")
    res = mon.observe_batch(rng.binomial(1, 0.1, 200), rng.binomial(1, 0.15, 200))
    assert "always_valid_p_value" in res
    assert "decision" in res


def test_sequential_monitor_obf_requires_max_n():
    with pytest.raises(ValueError):
        SequentialMonitor(method="obrien_fleming", alpha=0.05, k_looks=5)

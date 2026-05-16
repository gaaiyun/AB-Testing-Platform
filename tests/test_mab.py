"""mab.py 测试 — 验证三种策略的合理性。"""
import numpy as np
import pytest

from mab import EpsilonGreedy, ThompsonSampling, UCB1, simulate


def test_epsilon_greedy_pure_exploit_with_eps0():
    rng = np.random.default_rng(0)
    eg = EpsilonGreedy(["A", "B"], epsilon=0.0, rng=rng)
    eg.update("A", 1.0)
    eg.update("B", 0.0)
    for _ in range(20):
        assert eg.select() == "A"


def test_epsilon_greedy_pure_explore_with_eps1():
    rng = np.random.default_rng(0)
    eg = EpsilonGreedy(["A", "B", "C"], epsilon=1.0, rng=rng)
    eg.update("A", 1.0)
    pulls = {"A": 0, "B": 0, "C": 0}
    for _ in range(900):
        pulls[eg.select()] += 1
    # 完全随机时三个臂应大致均衡
    for c in pulls.values():
        assert 250 < c < 400


def test_ucb_pulls_each_arm_at_least_once_initially():
    rng = np.random.default_rng(0)
    u = UCB1(["A", "B", "C"], rng=rng)
    chosen = set()
    for _ in range(3):
        a = u.select()
        chosen.add(a)
        u.update(a, 0.5)
    assert chosen == {"A", "B", "C"}


def test_thompson_sampling_summary_has_posterior():
    ts = ThompsonSampling(["A", "B"], rng=np.random.default_rng(0))
    for _ in range(10):
        ts.update("A", 1.0)
        ts.update("B", 0.0)
    post = ts.posterior_summary()
    assert post["A"]["posterior_mean"] > post["B"]["posterior_mean"]
    lo, hi = post["A"]["credible_interval_95"]
    assert lo < post["A"]["posterior_mean"] < hi


def test_thompson_converges_to_best_arm():
    """跑 5000 步，Thompson sampling 应把绝大多数流量分到最优臂。"""
    rng = np.random.default_rng(0)
    ts = ThompsonSampling(["good", "bad"], rng=rng)
    result = simulate(ts, {"good": 0.30, "bad": 0.10}, n_steps=5000, rng=rng)
    assert result["summary"]["arms"]["good"]["share"] > 0.85
    assert result["optimal_arm"] == "good"


def test_ucb_beats_uniform_random_on_regret():
    rng = np.random.default_rng(0)
    ucb = UCB1(["A", "B"], rng=rng)
    eg = EpsilonGreedy(["A", "B"], epsilon=1.0, rng=np.random.default_rng(0))
    res_ucb = simulate(ucb, {"A": 0.5, "B": 0.2}, n_steps=2000, rng=np.random.default_rng(0))
    res_eg = simulate(eg, {"A": 0.5, "B": 0.2}, n_steps=2000, rng=np.random.default_rng(0))
    assert res_ucb["cumulative_regret"][-1] < res_eg["cumulative_regret"][-1]


def test_invalid_init_raises():
    with pytest.raises(ValueError):
        EpsilonGreedy(["only_one"])
    with pytest.raises(ValueError):
        EpsilonGreedy(["A", "B"], epsilon=1.5)
    with pytest.raises(ValueError):
        ThompsonSampling(["A", "B"], prior_alpha=0)


def test_update_unknown_arm_raises():
    eg = EpsilonGreedy(["A", "B"])
    with pytest.raises(KeyError):
        eg.update("X", 1.0)

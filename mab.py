"""
多臂老虎机（MAB）—— 自适应分流策略。

固定 A/B 测试的核心思路是「先全量探索，再全量采纳」。MAB 把"探索"和
"利用"持续混在一起：表现好的臂自动获得更多流量，表现差的臂自动萎缩。

适用场景
--------
- 业务后果可量化（点击率、收入），且能在短期内看到反馈
- 选项之间的"赢家"可能随时间变化（季节性、库存）
- 想最小化 regret（机会成本），而不是只回答"X 比 Y 好吗"

不适用场景
----------
- 决策需要可解释、可审计（医疗、金融监管） → 还是用固定 A/B 测试
- 处理效应有强延迟（订阅、留存）
- 不同变体之间存在 spillover

本模块实现三种最常用的策略：

1. EpsilonGreedy   — ε 概率随机探索、否则选当前最高均值
2. UCB1            — Upper Confidence Bound，理论 regret O(sqrt(n*log n))
3. ThompsonSampling — 用后验分布抽样，贝叶斯最优，工业最常用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np


@dataclass
class ArmStats:
    """单个臂的累计统计。"""

    name: str
    pulls: int = 0
    successes: float = 0.0  # 累计奖励（伯努利下就是成功数）
    sum_sq_reward: float = 0.0  # 用于连续奖励的方差估计

    @property
    def mean_reward(self) -> float:
        return self.successes / self.pulls if self.pulls > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.pulls < 2:
            return 1.0
        mean = self.mean_reward
        return max(self.sum_sq_reward / self.pulls - mean ** 2, 1e-12)


class _BaseBandit:
    """所有策略的公共接口。"""

    def __init__(self, arm_names: List[str], rng: Optional[np.random.Generator] = None):
        if len(arm_names) < 2:
            raise ValueError("至少需要 2 个臂")
        self.arms: Dict[str, ArmStats] = {name: ArmStats(name=name) for name in arm_names}
        self.rng = rng or np.random.default_rng()
        self.history: List[Dict] = []

    def select(self) -> str:
        raise NotImplementedError

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.arms:
            raise KeyError(f"未知 arm: {arm}")
        s = self.arms[arm]
        s.pulls += 1
        s.successes += float(reward)
        s.sum_sq_reward += float(reward) ** 2
        self.history.append({"arm": arm, "reward": float(reward), "t": sum(a.pulls for a in self.arms.values())})

    def summary(self) -> Dict:
        total_pulls = sum(a.pulls for a in self.arms.values())
        return {
            "total_pulls": total_pulls,
            "arms": {
                name: {
                    "pulls": s.pulls,
                    "share": s.pulls / total_pulls if total_pulls else 0.0,
                    "mean_reward": s.mean_reward,
                }
                for name, s in self.arms.items()
            },
            "best_arm": max(self.arms, key=lambda k: self.arms[k].mean_reward),
        }


class EpsilonGreedy(_BaseBandit):
    """ε-greedy：以 ε 概率随机探索，否则选当前最高均值臂。"""

    def __init__(self, arm_names: List[str], epsilon: float = 0.1, **kwargs):
        super().__init__(arm_names, **kwargs)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon 必须在 [0, 1]")
        self.epsilon = epsilon

    def select(self) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(self.arms.keys()))
        return max(self.arms, key=lambda k: self.arms[k].mean_reward)


class UCB1(_BaseBandit):
    """UCB1（Auer, Cesa-Bianchi, Fischer 2002）。"""

    def select(self) -> str:
        total = sum(a.pulls for a in self.arms.values())
        # 先把每个臂至少拉一次
        for name, s in self.arms.items():
            if s.pulls == 0:
                return name
        scores = {
            name: s.mean_reward + np.sqrt(2.0 * np.log(total) / s.pulls)
            for name, s in self.arms.items()
        }
        return max(scores, key=scores.get)


class ThompsonSampling(_BaseBandit):
    """
    Thompson sampling（伯努利奖励 → Beta 后验）。

    Parameters
    ----------
    arm_names : list[str]
    prior_alpha, prior_beta : float
        Beta 先验。1.0 / 1.0 = uniform；想强先验偏好"低 CTR"可以用 1.0 / 99.0。
    rng : np.random.Generator
    """

    def __init__(
        self,
        arm_names: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(arm_names, rng=rng)
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta 先验必须 > 0")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select(self) -> str:
        samples = {}
        for name, s in self.arms.items():
            a = self.prior_alpha + s.successes
            b = self.prior_beta + (s.pulls - s.successes)
            samples[name] = self.rng.beta(a, b)
        return max(samples, key=samples.get)

    def posterior_summary(self) -> Dict[str, Dict[str, float]]:
        """返回每个臂的 Beta 后验均值、CI。"""
        out = {}
        for name, s in self.arms.items():
            a = self.prior_alpha + s.successes
            b = self.prior_beta + (s.pulls - s.successes)
            mean = a / (a + b)
            # 95% credible interval (用 beta 分位数)
            from scipy.stats import beta as beta_dist  # noqa: WPS433
            ci = (float(beta_dist.ppf(0.025, a, b)), float(beta_dist.ppf(0.975, a, b)))
            out[name] = {"posterior_alpha": a, "posterior_beta": b,
                         "posterior_mean": mean, "credible_interval_95": ci}
        return out


# ----------------------------------------------------------------------------
# 模拟器：给定真实奖励概率，跑一段，输出 regret 与 share 演化
# ----------------------------------------------------------------------------

def simulate(
    strategy: _BaseBandit,
    true_rewards: Dict[str, float],
    n_steps: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Bernoulli 奖励模拟。

    Returns
    -------
    dict
        {'cumulative_regret': [...], 'share_over_time': {arm: [...]}, 'summary': ...}
    """
    rng = rng or np.random.default_rng()
    optimal_arm = max(true_rewards, key=true_rewards.get)
    optimal_reward = true_rewards[optimal_arm]

    cumulative_regret = []
    pulls_per_arm = {name: [] for name in true_rewards}
    cumulative = 0.0

    for _ in range(n_steps):
        arm = strategy.select()
        reward = float(rng.random() < true_rewards[arm])
        strategy.update(arm, reward)
        cumulative += (optimal_reward - true_rewards[arm])
        cumulative_regret.append(cumulative)
        total = sum(s.pulls for s in strategy.arms.values())
        for name in true_rewards:
            pulls_per_arm[name].append(strategy.arms[name].pulls / max(total, 1))

    return {
        "cumulative_regret": cumulative_regret,
        "share_over_time": pulls_per_arm,
        "summary": strategy.summary(),
        "optimal_arm": optimal_arm,
    }

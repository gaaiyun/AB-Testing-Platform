"""
序贯检验模块 — always-valid p-values 与 alpha-spending.

支持两类方法：

1. mSPRT（Mixture Sequential Probability Ratio Test）
   - 来源：Johari, Pekelis, Walsh (2015), "Always Valid Inference: Continuous
     Monitoring of A/B Tests" (Optimizely).
   - 特点：可在任意时刻"偷看"实验结果而不会膨胀第一类错误率。
   - 用法：在线 update，每个时间点都能拿到 always-valid p-value 和
     always-valid 置信区间。

2. Alpha-Spending（O'Brien-Fleming / Pocock 边界）
   - 来源：Lan & DeMets (1983).
   - 特点：在预设的"中期分析"次数下分配 alpha 预算，给出每一次允许的
     拒绝阈值。

两类方法对应不同业务场景：

- 流量持续累积、想随时看结果 → mSPRT
- 有固定的中期检查点（如 25%/50%/100%） → Alpha-Spending

⚠️ 不要用裸的 z 检验做"达到显著就停"——这会严重膨胀假阳性率（详见
`docs/COMMON_PITFALLS.md`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# 1. mSPRT
# ----------------------------------------------------------------------------

@dataclass
class MSPRTState:
    """mSPRT 在线状态。"""

    n_control: int = 0
    n_treatment: int = 0
    sum_control: float = 0.0
    sum_treatment: float = 0.0
    sum_sq_control: float = 0.0
    sum_sq_treatment: float = 0.0

    # 当前 always-valid p-value 的下确界（即历史最小值）
    min_p_value: float = 1.0

    history: List[Tuple[int, float, float, float]] = field(default_factory=list)
    """每次 update 后追加一条 (n_total, mean_diff, av_p_value, log_lik_ratio)."""


class MSPRT:
    """
    Mixture SPRT，用 N(0, tau²) 作为效应的混合先验。

    Parameters
    ----------
    tau : float
        混合先验的标准差，反映对效应大小的先验。经验值：取业务可接受的
        MDE（最小可检测效应），默认 0.01（绝对差异 1pp）。
    alpha : float
        显著性水平（用于推 always-valid 阈值；mSPRT 的 p 值本身在任意停止
        时间都不会膨胀，alpha 只用于"现在能不能拒绝 H0"的判断）。
    metric : {"proportion", "mean"}
        指标类型：转化率（伯努利）或连续指标（高斯近似）。

    Notes
    -----
    Always-valid 性质：对任意停止时间 N（哪怕是数据相关的），有
        P_{H0}( min_{n <= N} p_n <= alpha ) <= alpha.
    所以可以反复"偷看"、随时决定是否停止，而不会损害第一类错误率。
    """

    def __init__(
        self,
        tau: float = 0.01,
        alpha: float = 0.05,
        metric: Literal["proportion", "mean"] = "proportion",
    ) -> None:
        if tau <= 0:
            raise ValueError("tau 必须 > 0")
        if not 0 < alpha < 1:
            raise ValueError("alpha 必须在 (0, 1) 内")
        if metric not in ("proportion", "mean"):
            raise ValueError("metric 必须是 'proportion' 或 'mean'")

        self.tau = tau
        self.alpha = alpha
        self.metric = metric
        self.state = MSPRTState()

    def update_proportion(self, control_successes: int, treatment_successes: int,
                          control_n: int, treatment_n: int) -> dict:
        """以累计计数 update（适合从汇总数据回放）。"""
        if self.metric != "proportion":
            raise ValueError("当前实例不是 proportion 模式")
        self.state.sum_control = float(control_successes)
        self.state.sum_treatment = float(treatment_successes)
        self.state.n_control = int(control_n)
        self.state.n_treatment = int(treatment_n)
        return self._evaluate()

    def add_observation(self, control: float, treatment: float) -> dict:
        """逐条样本流入（每个时间点同时观测到一对 control/treatment）。"""
        self.state.sum_control += float(control)
        self.state.sum_treatment += float(treatment)
        self.state.sum_sq_control += float(control) ** 2
        self.state.sum_sq_treatment += float(treatment) ** 2
        self.state.n_control += 1
        self.state.n_treatment += 1
        return self._evaluate()

    def add_batch(self, control: Iterable[float], treatment: Iterable[float]) -> dict:
        """成批添加并在批末输出一次评估。"""
        c = np.asarray(list(control), dtype=float)
        t = np.asarray(list(treatment), dtype=float)
        if len(c) != len(t):
            raise ValueError("control/treatment 批长度需相同（双臂同步流入假设）")
        self.state.sum_control += c.sum()
        self.state.sum_treatment += t.sum()
        self.state.sum_sq_control += (c ** 2).sum()
        self.state.sum_sq_treatment += (t ** 2).sum()
        self.state.n_control += len(c)
        self.state.n_treatment += len(t)
        return self._evaluate()

    # ------------------------------------------------------------------
    # 核心：计算 likelihood ratio 与 always-valid p-value
    # ------------------------------------------------------------------
    def _evaluate(self) -> dict:
        n_c = self.state.n_control
        n_t = self.state.n_treatment
        if min(n_c, n_t) < 2:
            return self._empty_result()

        mean_c = self.state.sum_control / n_c
        mean_t = self.state.sum_treatment / n_t
        diff = mean_t - mean_c

        if self.metric == "proportion":
            # 两个伯努利的差的方差近似：p(1-p) 加性
            p_pool = (self.state.sum_control + self.state.sum_treatment) / (n_c + n_t)
            sigma2 = p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t)
        else:
            var_c = max(
                (self.state.sum_sq_control - n_c * mean_c ** 2) / max(n_c - 1, 1),
                1e-12,
            )
            var_t = max(
                (self.state.sum_sq_treatment - n_t * mean_t ** 2) / max(n_t - 1, 1),
                1e-12,
            )
            sigma2 = var_c / n_c + var_t / n_t

        # mSPRT 似然比（mixture prior N(0, tau²)）
        # Λ_n = sqrt(sigma2/(sigma2 + n*tau²)) * exp( diff² / (2*(sigma2/n_eff + tau² ?)... )
        # 这里用 effective n 形式：将差的分布看作 N(delta, sigma2)，先验 delta ~ N(0, tau²)。
        # 经典推导见 Johari et al. (2015) eq.(2)。
        tau2 = self.tau ** 2
        # 注意 sigma2 已经是 diff 的方差（不含 n），因此似然比的指数项分母用 sigma2 + tau²
        # 但要保持 alpha 控制，我们用 Robbins (1970) 形式：
        log_lr = 0.5 * np.log(sigma2 / (sigma2 + tau2)) + \
                 (diff ** 2) * tau2 / (2.0 * sigma2 * (sigma2 + tau2))
        lr = float(np.exp(log_lr))

        av_p = min(1.0, 1.0 / lr) if lr > 0 else 1.0
        self.state.min_p_value = min(self.state.min_p_value, av_p)

        # always-valid 置信区间：求满足 |delta_0| 使似然比 <= 1/alpha 的边界
        # 等价于：|diff - delta_0|² <= 2*sigma2*(sigma2+tau²)/tau² * (log(1/alpha)+0.5*log((sigma2+tau²)/sigma2))
        log_thresh = np.log(1.0 / self.alpha)
        rhs = 2.0 * sigma2 * (sigma2 + tau2) / tau2 * (
            log_thresh + 0.5 * np.log((sigma2 + tau2) / sigma2)
        )
        half_width = float(np.sqrt(max(rhs, 0.0)))
        ci = (diff - half_width, diff + half_width)

        self.state.history.append((n_c + n_t, diff, av_p, log_lr))

        return {
            "n_total": n_c + n_t,
            "diff": diff,
            "mean_control": mean_c,
            "mean_treatment": mean_t,
            "log_likelihood_ratio": log_lr,
            "always_valid_p_value": av_p,
            "running_min_p_value": self.state.min_p_value,
            "always_valid_ci": ci,
            "decision": self.decide(av_p),
        }

    def decide(self, av_p: Optional[float] = None) -> str:
        """给出可读的决策建议。"""
        p = av_p if av_p is not None else self.state.min_p_value
        if p <= self.alpha:
            return "reject_null"
        # 还在采集中
        return "continue"

    def _empty_result(self) -> dict:
        return {
            "n_total": self.state.n_control + self.state.n_treatment,
            "diff": 0.0,
            "mean_control": 0.0,
            "mean_treatment": 0.0,
            "log_likelihood_ratio": 0.0,
            "always_valid_p_value": 1.0,
            "running_min_p_value": 1.0,
            "always_valid_ci": (-np.inf, np.inf),
            "decision": "continue",
        }


# ----------------------------------------------------------------------------
# 2. Alpha-Spending
# ----------------------------------------------------------------------------

@dataclass
class AlphaSpendingBoundary:
    """一次预设的 alpha-spending 边界。"""

    method: str
    alpha: float
    information_fractions: List[float]
    z_boundaries: List[float]

    def stop(self, observed_z: float, information_fraction: float) -> bool:
        """在给定 information_fraction（即 n_current/n_max）下是否应当拒绝。"""
        # 选最接近且 <= 当前 information_fraction 的边界
        applicable = [
            (frac, z)
            for frac, z in zip(self.information_fractions, self.z_boundaries)
            if frac <= information_fraction + 1e-9
        ]
        if not applicable:
            return False
        _, z_b = applicable[-1]
        return abs(observed_z) >= z_b


def obrien_fleming_boundary(alpha: float, k_looks: int) -> AlphaSpendingBoundary:
    """
    O'Brien-Fleming 边界：前期严格、末期接近常规 alpha。

    使用 Lan-DeMets 近似的 alpha-spending 函数：
        f(t) = 2 - 2 * Phi(z_{alpha/2} / sqrt(t))
    在 information_fractions = 1/k_looks, 2/k_looks, ..., 1 处分配 alpha。

    Parameters
    ----------
    alpha : float
        总 type-I 错误预算（双尾）。
    k_looks : int
        计划的中期分析次数（包括最终一次）。

    Returns
    -------
    AlphaSpendingBoundary
        每次中期分析对应的 |Z| 拒绝阈值。
    """
    if k_looks < 1:
        raise ValueError("k_looks 必须 >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须在 (0, 1) 内")

    fractions = np.arange(1, k_looks + 1) / k_looks
    z_alpha_half = stats.norm.ppf(1 - alpha / 2)

    # Lan-DeMets 近似 spending 函数
    cum_alpha_spent = 2.0 - 2.0 * stats.norm.cdf(z_alpha_half / np.sqrt(fractions))
    # 每个 look 新花费的 alpha
    incremental = np.diff(np.concatenate([[0.0], cum_alpha_spent]))
    # 把每次的 incremental alpha 转成 z 阈值（双尾）
    z_bounds = [stats.norm.ppf(1 - inc / 2) if inc > 0 else 8.0 for inc in incremental]

    return AlphaSpendingBoundary(
        method="O'Brien-Fleming (Lan-DeMets)",
        alpha=alpha,
        information_fractions=fractions.tolist(),
        z_boundaries=z_bounds,
    )


def pocock_boundary(alpha: float, k_looks: int) -> AlphaSpendingBoundary:
    """
    Pocock 边界：每次 look 用相同的 z 阈值，便于沟通。

    Pocock 没有解析形式，但有近似拟合（Pocock 1977 表 + 多元正态精确计算）。
    这里用经验近似（Wassmer & Brannath, 2016 表 2.3，相对误差 < 1%）。
    """
    if k_looks < 1:
        raise ValueError("k_looks 必须 >= 1")

    pocock_table = {
        # alpha=0.05 双尾
        0.05: {1: 1.960, 2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413,
               6: 2.453, 8: 2.512, 10: 2.555},
        0.01: {1: 2.576, 2: 2.772, 3: 2.873, 4: 2.939, 5: 2.986,
               6: 3.023, 8: 3.078, 10: 3.117},
    }
    if alpha not in pocock_table:
        raise ValueError(
            f"暂只内置 alpha ∈ {sorted(pocock_table.keys())} 的 Pocock 表，"
            f"其他值请用 obrien_fleming_boundary 或自行扩展表"
        )
    table = pocock_table[alpha]
    # 找最接近的 k_looks
    available = sorted(table.keys())
    if k_looks in table:
        z = table[k_looks]
    else:
        # 线性插值
        lower = max(k for k in available if k < k_looks) if any(k < k_looks for k in available) else available[0]
        upper = min(k for k in available if k > k_looks) if any(k > k_looks for k in available) else available[-1]
        if lower == upper:
            z = table[lower]
        else:
            z = table[lower] + (table[upper] - table[lower]) * (k_looks - lower) / (upper - lower)

    fractions = (np.arange(1, k_looks + 1) / k_looks).tolist()
    z_bounds = [z] * k_looks

    return AlphaSpendingBoundary(
        method=f"Pocock (constant z={z:.3f})",
        alpha=alpha,
        information_fractions=fractions,
        z_boundaries=z_bounds,
    )


# ----------------------------------------------------------------------------
# 3. 在线监控（顶层 API）
# ----------------------------------------------------------------------------

class SequentialMonitor:
    """
    把 mSPRT 与 alpha-spending 包成一个统一的"实验监控器"。

    使用流程
    --------
    >>> mon = SequentialMonitor(method="msprt", tau=0.01, alpha=0.05)
    >>> for day in range(1, 15):
    ...     control_today = np.random.binomial(1, 0.1, 1000)
    ...     treatment_today = np.random.binomial(1, 0.11, 1000)
    ...     res = mon.observe_batch(control_today, treatment_today)
    ...     if res["decision"] == "reject_null":
    ...         print(f"day {day}: 已可拒绝 H0，p={res['always_valid_p_value']:.4f}")
    ...         break
    """

    def __init__(
        self,
        method: Literal["msprt", "obrien_fleming", "pocock"] = "msprt",
        tau: float = 0.01,
        alpha: float = 0.05,
        k_looks: int = 5,
        max_n_per_arm: Optional[int] = None,
        metric: Literal["proportion", "mean"] = "proportion",
    ) -> None:
        self.method = method
        self.alpha = alpha
        if method == "msprt":
            self.msprt = MSPRT(tau=tau, alpha=alpha, metric=metric)
            self.boundary: Optional[AlphaSpendingBoundary] = None
        else:
            self.msprt = None
            if max_n_per_arm is None:
                raise ValueError("alpha-spending 需要预设 max_n_per_arm")
            self.boundary = (
                obrien_fleming_boundary(alpha, k_looks)
                if method == "obrien_fleming"
                else pocock_boundary(alpha, k_looks)
            )
        self.max_n_per_arm = max_n_per_arm
        self.metric = metric
        self.n_control = 0
        self.n_treatment = 0
        self.sum_control = 0.0
        self.sum_treatment = 0.0

    def observe_batch(self, control: np.ndarray, treatment: np.ndarray) -> dict:
        if self.method == "msprt":
            return self.msprt.add_batch(control, treatment)

        # alpha-spending 路径
        c = np.asarray(control, dtype=float)
        t = np.asarray(treatment, dtype=float)
        self.sum_control += c.sum()
        self.sum_treatment += t.sum()
        self.n_control += len(c)
        self.n_treatment += len(t)

        info_frac = max(self.n_control, self.n_treatment) / float(self.max_n_per_arm)
        info_frac = min(info_frac, 1.0)

        if self.metric == "proportion":
            mean_c = self.sum_control / self.n_control
            mean_t = self.sum_treatment / self.n_treatment
            p_pool = (self.sum_control + self.sum_treatment) / (self.n_control + self.n_treatment)
            se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / self.n_control + 1 / self.n_treatment)))
        else:
            mean_c = self.sum_control / self.n_control
            mean_t = self.sum_treatment / self.n_treatment
            # 简化：用同方差近似（生产环境建议用 Welch）
            se = float(np.sqrt(2.0 / self.n_control))  # placeholder, 调用方应传方差
        diff = mean_t - mean_c
        z = diff / se if se > 0 else 0.0
        stop = self.boundary.stop(z, info_frac)
        return {
            "n_total": self.n_control + self.n_treatment,
            "diff": diff,
            "z_stat": z,
            "information_fraction": info_frac,
            "boundary_method": self.boundary.method,
            "decision": "reject_null" if stop else "continue",
        }

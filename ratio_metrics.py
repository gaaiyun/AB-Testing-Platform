"""Ratio metrics 的 delta method z-test。

实际 A/B 实验里"每用户平均订单数"、"转化率（每 session）"这类"指标 =
numerator / denominator"非常常见（ARPU、AOV、CTR、人均订单数等）。

普通 z-test 把这些指标当成"均值"处理是错的，因为：

1. 用户层面：每个用户产生若干 sessions / orders / 浏览，每个事件是一个
   numerator 贡献，但同一个用户的多次事件不独立。
2. 实验单位（user）与 metric 单位（session）不一致 —— 直接对 session
   层面 t-test 等于把每个用户算作 N 次，独立样本假设被破坏。
3. 即使按用户聚合，"X̄ / Ȳ" 的标准误不是 SE(X̄) / SE(Ȳ)，而是要做
   delta method 一阶 Taylor 展开。

Delta method（一阶近似）：

    Var(X/Y) ≈ (1/μ_Y)² · Var(X) + (μ_X/μ_Y²)² · Var(Y)
               - 2 · (μ_X/μ_Y³) · Cov(X, Y)

把每个用户的 (X_i, Y_i) 视为独立样本 → 用样本估计 μ_X, μ_Y, Var(X),
Var(Y), Cov(X, Y) 代入。

参考：
- Deng, Knoblich, Lu (2018) "Applying the Delta method in metric
  analytics", Microsoft Experimentation Platform. (经典 reference)
- Kohavi, Tang, Xu (2020) *Trustworthy Online Controlled Experiments*,
  Cambridge UP. Chapter 18 "The Delta Method".
- Hsiao et al. (2018, KDD) — Yandex 同款方法的工程化讨论。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class RatioStats:
    """单组 ratio metric 的描述统计。"""
    mean_x: float          # numerator 样本均值
    mean_y: float          # denominator 样本均值
    ratio: float           # X̄ / Ȳ
    var_ratio: float       # delta method 估计的 Var(X̄/Ȳ)
    se_ratio: float        # sqrt(var)
    n: int                 # 样本量（用户数）

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class RatioTestResult:
    """两组 ratio metric 对比的检验结果。"""
    control: RatioStats
    treatment: RatioStats
    diff_ratio: float                            # treatment.ratio - control.ratio
    relative_lift_pct: float                     # diff / control.ratio × 100
    se_diff: float                               # SE 差（两个独立组的 Var 相加再开方）
    z_statistic: float
    p_value: float                               # 两尾
    significant: bool
    confidence_interval: Tuple[float, float]     # 差值的置信区间
    alpha: float
    additional_info: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "control": self.control.to_dict(),
            "treatment": self.treatment.to_dict(),
            "diff_ratio": float(self.diff_ratio),
            "relative_lift_pct": float(self.relative_lift_pct),
            "se_diff": float(self.se_diff),
            "z_statistic": float(self.z_statistic),
            "p_value": float(self.p_value),
            "significant": bool(self.significant),
            "confidence_interval": [float(self.confidence_interval[0]),
                                     float(self.confidence_interval[1])],
            "alpha": float(self.alpha),
            "additional_info": dict(self.additional_info),
        }


# --- 单组 delta-method 估计 ----------------------------------------------

def delta_method_variance(
    numerator: Sequence[float],
    denominator: Sequence[float],
) -> RatioStats:
    """对一组样本 (X_i, Y_i)，用 delta method 估 X̄/Ȳ 的 Variance。

    每个 i 是一个 *实验单位*（通常是 user）。X_i 是该用户的 numerator
    （如订单数），Y_i 是 denominator（如 sessions 数）。

    Raises
    ------
    ValueError : 长度不一致 / 样本太少 / Ȳ ≈ 0
    """
    x = np.asarray(numerator, dtype=float)
    y = np.asarray(denominator, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"numerator / denominator 长度不一致：{x.shape} vs {y.shape}")
    if len(x) < 2:
        raise ValueError("样本量 < 2 无法估方差")
    n = len(x)

    mu_x = float(x.mean())
    mu_y = float(y.mean())
    if abs(mu_y) < 1e-12:
        raise ValueError("denominator 样本均值 ≈ 0，ratio 无定义")

    # 样本方差（ddof=1）
    var_x = float(x.var(ddof=1))
    var_y = float(y.var(ddof=1))
    # 样本协方差
    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])

    # delta method（per-observation）→ X̄/Ȳ 方差 = 单观测方差 / n
    # Var(X̄/Ȳ) = 1/n × [ Var(X)/μ_Y² - 2 μ_X Cov(X,Y) / μ_Y³ + μ_X² Var(Y) / μ_Y⁴ ]
    inv_muy2 = 1.0 / (mu_y ** 2)
    var_ratio = (1.0 / n) * (
        var_x * inv_muy2
        - 2.0 * mu_x * cov_xy / (mu_y ** 3)
        + (mu_x ** 2) * var_y / (mu_y ** 4)
    )
    var_ratio = max(var_ratio, 0.0)
    se_ratio = math.sqrt(var_ratio)
    ratio = mu_x / mu_y

    return RatioStats(
        mean_x=mu_x, mean_y=mu_y, ratio=ratio,
        var_ratio=var_ratio, se_ratio=se_ratio, n=n,
    )


# --- 两组对比检验 -----------------------------------------------------

def ratio_metric_test(
    control_numerator: Sequence[float],
    control_denominator: Sequence[float],
    treatment_numerator: Sequence[float],
    treatment_denominator: Sequence[float],
    alpha: float = 0.05,
) -> RatioTestResult:
    """两组 ratio metric 的 delta-method z-test。

    Parameters
    ----------
    *_numerator : 每个用户的 X 序列
    *_denominator : 每个用户的 Y 序列
    alpha : 显著性水平

    Notes
    -----
    假设两组用户独立 → Var(Δ) = Var(R_t) + Var(R_c)。
    """
    c = delta_method_variance(control_numerator, control_denominator)
    t = delta_method_variance(treatment_numerator, treatment_denominator)

    diff = t.ratio - c.ratio
    var_diff = c.var_ratio + t.var_ratio
    se_diff = math.sqrt(var_diff) if var_diff > 0 else 0.0
    if se_diff > 1e-12:
        z = diff / se_diff
        p_two_sided = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    elif abs(diff) <= 1e-12:
        z = 0.0
        p_two_sided = 1.0
    else:
        z = math.copysign(math.inf, diff)
        p_two_sided = 0.0

    # 双尾 CI
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_alpha * se_diff
    ci_high = diff + z_alpha * se_diff

    if abs(c.ratio) > 1e-12:
        rel_lift = (diff / c.ratio) * 100.0
    else:
        rel_lift = 0.0

    return RatioTestResult(
        control=c, treatment=t,
        diff_ratio=diff, relative_lift_pct=rel_lift,
        se_diff=se_diff, z_statistic=z, p_value=float(p_two_sided),
        significant=p_two_sided < alpha,
        confidence_interval=(ci_low, ci_high),
        alpha=alpha,
    )


# --- 与"错误用法"对比的诊断 -----------------------------------------

def naive_event_level_test(
    control_event_values: Sequence[float],
    treatment_event_values: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """v1 模式：把每条 event 当成独立样本做 t-test。

    这是大部分新手 A/B 工具的 default。返回 (t_stat, p_value)。
    与 ``ratio_metric_test`` 对比：前者 p-value 通常更小（假阳性更多），
    因为忽略了 user-level clustering。

    用法：把两组的 event-level numerator 平铺成长 list 喂进来即可。
    """
    c = np.asarray(control_event_values, dtype=float)
    t = np.asarray(treatment_event_values, dtype=float)
    result = stats.ttest_ind(c, t, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def required_sample_size_delta(
    baseline_ratio: float,
    minimum_detectable_lift_pct: float,
    cv_x: float,
    cv_y: float,
    correlation_xy: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """基于 delta method 的样本量计算（每组）。

    一阶近似：要检测 ``minimum_detectable_lift_pct`` 的相对提升，CV =
    σ/μ 形式表达噪声。

    n ≈ ((z_α + z_β)² × Var(R)/R²) / (MDE)²

    其中 Var(R)/R² 的 delta 展开：CV_x² + CV_y² - 2 ρ CV_x CV_y
    （单观测层面的方差除以 R²）。

    Parameters
    ----------
    baseline_ratio : 当前 X/Y 比值
    minimum_detectable_lift_pct : 最小可检相对提升百分点（5% 写 0.05）
    cv_x : X 的变异系数 σ_x / μ_x
    cv_y : Y 的变异系数
    correlation_xy : X, Y 相关系数
    alpha / power : 显著性水平 / 检验力
    """
    if minimum_detectable_lift_pct <= 0:
        raise ValueError("MDE 必须 > 0")
    if not -1 <= correlation_xy <= 1:
        raise ValueError("correlation_xy 必须 ∈ [-1, 1]")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Var(R)/R² ≈ CV_x² + CV_y² - 2ρ CV_x CV_y（单观测）
    var_over_r_sq = cv_x ** 2 + cv_y ** 2 - 2 * correlation_xy * cv_x * cv_y
    var_over_r_sq = max(var_over_r_sq, 1e-12)

    # 两组独立 → Var(Δ) = 2 × Var(R)/n
    n = ((z_alpha + z_beta) ** 2 * 2 * var_over_r_sq) / (minimum_detectable_lift_pct ** 2)
    return int(math.ceil(n))

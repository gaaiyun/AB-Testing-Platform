"""
CUPED — Controlled-experiment Using Pre-Experiment Data.

来源：Deng, Xu, Kohavi, Walker (2013), "Improving the Sensitivity of Online
Controlled Experiments by Utilizing Pre-Experiment Data" (Microsoft).

原理
----
传统 A/B 检验直接比较 Y_t、Y_c。如果有一个与 Y 高度相关的 pre-experiment
协变量 X（比如实验前 7 天的人均订单数、人均收入），就可以做调整：

    Y' = Y - θ * (X - E[X]),    θ = Cov(X, Y) / Var(X)

这个 Y' 的均值与 Y 相同（因为 E[X - E[X]] = 0，θ 也是从全样本估计的），
但**方差被压缩到 Var(Y) * (1 - ρ²)**，其中 ρ 是 X 与 Y 的相关系数。

效果
----
- 方差缩减 = 1 - ρ²。例如 ρ = 0.5 → 减少 25% 方差 → 同 power 下样本量减
  25%；ρ = 0.7 → 减少 49% 方差 → 样本量减约一半。
- 在 Microsoft、Netflix、Booking 等都被作为标配。

注意事项
--------
1. **θ 必须从所有样本（合并 control+treatment）估计**，否则会引入偏倚。
2. **X 必须是 pre-experiment** 测得的——如果 X 是实验期间数据，可能被处理
   影响，CUPED 会偏。
3. 缺失 X 的样本要么剔除、要么用 X = 全样本均值（不影响 CUPED 调整后的
   效应估计的无偏性，但削弱方差缩减）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class CupedResult:
    """CUPED 调整后的两样本比较结果。"""

    theta: float
    correlation: float
    variance_reduction: float  # 1 - (Var(Y')/Var(Y))
    adjusted_mean_control: float
    adjusted_mean_treatment: float
    adjusted_diff: float
    se_diff: float
    t_stat: float
    p_value: float
    ci_95: Tuple[float, float]
    n_control: int
    n_treatment: int


def estimate_theta(x: np.ndarray, y: np.ndarray) -> float:
    """估计 θ = Cov(X, Y) / Var(X)（全样本，未分组）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x、y 必须等长")
    var_x = np.var(x, ddof=1)
    if var_x <= 0:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def cuped_adjust(
    x: np.ndarray,
    y: np.ndarray,
    theta: Optional[float] = None,
    x_global_mean: Optional[float] = None,
) -> np.ndarray:
    """
    对 Y 做 CUPED 调整。

    Parameters
    ----------
    x : array-like
        pre-experiment 协变量。
    y : array-like
        实验期观测指标。
    theta : float, optional
        若给定就直接用；否则从 (x, y) 估计。
    x_global_mean : float, optional
        E[X] 的估计；缺省用 x 的均值（应该传入 control+treatment 合并后的
        均值才正确，参见 cuped_two_sample）。

    Returns
    -------
    np.ndarray
        调整后的 Y'。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if theta is None:
        theta = estimate_theta(x, y)
    if x_global_mean is None:
        x_global_mean = float(np.mean(x))
    return y - theta * (x - x_global_mean)


def cuped_two_sample(
    x_control: np.ndarray,
    y_control: np.ndarray,
    x_treatment: np.ndarray,
    y_treatment: np.ndarray,
    alpha: float = 0.05,
) -> CupedResult:
    """
    用 CUPED 做两样本 t 检验（推荐入口）。

    返回方差缩减率、调整后的均值差、p-value、95% CI 等。

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 5000
    >>> x_c = rng.normal(10, 3, n); x_t = rng.normal(10, 3, n)
    >>> y_c = x_c * 0.7 + rng.normal(0, 2, n)          # ρ ≈ 0.72
    >>> y_t = x_t * 0.7 + rng.normal(0.3, 2, n)        # 真实效应 0.3
    >>> r = cuped_two_sample(x_c, y_c, x_t, y_t)
    >>> r.variance_reduction > 0.4
    True
    """
    x_c = np.asarray(x_control, dtype=float)
    y_c = np.asarray(y_control, dtype=float)
    x_t = np.asarray(x_treatment, dtype=float)
    y_t = np.asarray(y_treatment, dtype=float)

    if len(x_c) != len(y_c):
        raise ValueError("control 的 x、y 长度需相同")
    if len(x_t) != len(y_t):
        raise ValueError("treatment 的 x、y 长度需相同")

    # 1. 在合并样本上估 θ
    x_all = np.concatenate([x_c, x_t])
    y_all = np.concatenate([y_c, y_t])
    theta = estimate_theta(x_all, y_all)
    x_global_mean = float(np.mean(x_all))

    # 2. 调整
    y_c_adj = cuped_adjust(x_c, y_c, theta=theta, x_global_mean=x_global_mean)
    y_t_adj = cuped_adjust(x_t, y_t, theta=theta, x_global_mean=x_global_mean)

    # 3. 方差缩减率
    var_y = np.var(y_all, ddof=1)
    var_y_adj = np.var(np.concatenate([y_c_adj, y_t_adj]), ddof=1)
    var_reduction = max(0.0, 1.0 - var_y_adj / var_y) if var_y > 0 else 0.0
    corr = float(np.corrcoef(x_all, y_all)[0, 1]) if len(x_all) > 1 else 0.0

    # 4. Welch's t-test on adjusted
    n_c, n_t = len(y_c_adj), len(y_t_adj)
    mean_c = float(np.mean(y_c_adj))
    mean_t = float(np.mean(y_t_adj))
    var_c = float(np.var(y_c_adj, ddof=1))
    var_t = float(np.var(y_t_adj, ddof=1))
    diff = mean_t - mean_c
    se = float(np.sqrt(var_c / n_c + var_t / n_t))
    if se == 0:
        t_stat = 0.0
        p_value = 1.0
        ci = (diff, diff)
    else:
        t_stat = diff / se
        df = (var_c / n_c + var_t / n_t) ** 2 / (
            (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
        )
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        ci = (diff - t_crit * se, diff + t_crit * se)

    return CupedResult(
        theta=theta,
        correlation=corr,
        variance_reduction=var_reduction,
        adjusted_mean_control=mean_c,
        adjusted_mean_treatment=mean_t,
        adjusted_diff=diff,
        se_diff=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_95=ci,
        n_control=n_c,
        n_treatment=n_t,
    )


def estimate_sample_size_savings(correlation: float) -> float:
    """给定预期 ρ，返回相对裸 t-test 的样本量节省比例（0~1）。"""
    return float(min(max(correlation ** 2, 0.0), 0.99))

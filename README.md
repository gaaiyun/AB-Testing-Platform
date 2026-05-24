# A/B 测试分析平台

> 把 A/B 测试做对 — 不只是会算 z 检验。

这个项目从 v1 的"教科书统计实现 + Streamlit"演进到 v2 的"工业级 A/B 工具箱"。

v2 引入了三个**别处不容易凑齐的功能**：

| 模块 | 解决什么问题 | 参考文献 |
|---|---|---|
| **mSPRT** (`sequential.py`) | 让你随时偷看实验结果而不膨胀假阳性率 | [Johari, Pekelis, Walsh (2015)](https://arxiv.org/pdf/1512.04922) |
| **CUPED** (`cuped.py`) | 用 pre-experiment 协变量减 30%~50% 样本量 | [Deng, Xu, Kohavi, Walker (2013)](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf) |
| **Multi-Armed Bandit** (`mab.py`) | 自适应分流，最小化机会成本 | UCB1 / Thompson sampling |

v3 第一项已交付：

| 模块 | 解决什么问题 | 参考文献 |
|---|---|---|
| **Ratio metrics delta method** (`ratio_metrics.py`) | 人均订单数 / ARPU / CTR 这类"X̄/Ȳ"指标的正确 SE 估计 + 样本量计算 | [Deng, Knoblich, Lu (2018) — Microsoft Experimentation Platform](https://www.exp-platform.com/Documents/2018-DengKnoblichLu-Delta_Method.pdf); Kohavi/Tang/Xu (2020) Ch. 18 |

外加一个**修复**：v1 在 dashboard "序贯分析" 页面里直接写了 `if p < 0.05: 停止` —— 这是 A/B 测试领域**最经典的错误**之一（peeking 会把真实假阳性率从 5% 拉到 20%+）。v2 把这段替换成 mSPRT。

---

## 快速开始

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

或者直接以库的方式调用：

```python
from sequential import MSPRT

mon = MSPRT(tau=0.01, alpha=0.05, metric="proportion")
for day in range(1, 15):
    control_today, treatment_today = collect_today()      # 你的数据源
    res = mon.add_batch(control_today, treatment_today)
    print(f"day {day}: AV p-value = {res['always_valid_p_value']:.4f}")
    if res["decision"] == "reject_null":
        break
```

```python
from cuped import cuped_two_sample

# 用实验前 7 天的人均订单数 X 给当前指标 Y 做调整
result = cuped_two_sample(x_control, y_control, x_treatment, y_treatment)
print(f"方差缩减率: {result.variance_reduction:.1%}")
print(f"调整后 p-value: {result.p_value:.4f}")
```

```python
from mab import ThompsonSampling

ts = ThompsonSampling(["control", "treatment_A", "treatment_B"])
for i in range(10_000):
    arm = ts.select()
    reward = serve_and_observe(arm)                       # 你的业务回调
    ts.update(arm, reward)
print(ts.summary())
```

---

## 模块清单

```
ab-testing-platform/
├─ statistical_test.py     # 频率学派 + Bayesian 单次检验（含 Welch's t 修复）
├─ experiment_designer.py  # 样本量、power、流量分配
├─ sequential.py           # mSPRT + alpha-spending（O'Brien-Fleming / Pocock）
├─ cuped.py                # CUPED 方差缩减
├─ mab.py                  # ε-greedy / UCB1 / Thompson sampling + simulator
├─ result_visualizer.py    # plotly 可视化
├─ report.py               # 自包含 HTML 报告导出
└─ dashboard.py            # Streamlit 6 个页面（实验设计 / 数据导入 / 统计检验 /
                          #   可视化 / 序贯分析 / 实验报告）
```

测试在 `tests/` 下，可直接 `pytest -o addopts="" tests/` 跑（默认开启 coverage，关掉就行）。

---

## 决策指引

**裸 z 检验什么时候够用？**
固定样本量、固定截止日期、只在到期后看一次结果。绝大多数业务场景**不属于这一类**。

**什么时候用 mSPRT？**
流量持续累积，团队想随时看仪表盘并决定是否提前停止。`tau` 设得越小（先验越窄）= 早期越保守。**经验值**：`tau = 你认为有业务意义的最小相对效应`。

**什么时候用 alpha-spending（O'Brien-Fleming）？**
你有明确的"中期检查点"（比如周一例会看 4 次），并且想给统计学家一个干净的报告。O'Brien-Fleming 早期严格、末期接近常规 alpha，是医疗 / 监管最常用的选择。

**什么时候用 CUPED？**
你能找到一个 ρ ≥ 0.3 的 pre-experiment 协变量（实验开始前就已经测得、且与目标指标强相关）。例如：
- 目标指标 = 实验期人均订单数 → 协变量 = 实验前 14 天人均订单数
- 目标指标 = 留存率 → 协变量 = 前 30 天活跃天数
- 目标指标 = 转化金额 → 协变量 = 前 90 天 LTV

ρ = 0.5 大约能省 25% 样本，ρ = 0.7 大约能省一半。

**什么时候用 MAB？**
- ✅ 业务回报可量化、反馈及时（点击、加购）
- ✅ 你愿意接受"实验过程中流量倾斜"
- ❌ 决策需要可审计 / 监管批准
- ❌ 反馈延迟很久（订阅、长期留存）

---

## 与 v1 的差异

v1 的代码并没有删 — 频率学派、Bayesian、多重检验校正、可视化都还在原位。v2 是**增量**：

- ✅ 新增四个模块（sequential / cuped / mab / report）
- ✅ 修复 `statistical_test.py:153-154` Welch's t-test 的 `UnboundLocalError`（v1 进 else 分支必崩）
- ✅ 修复 `dashboard.py` 序贯分析页的统计错误（详见 `docs/COMMON_PITFALLS.md`）
- ✅ 32 个新测试，含 100-trial 的 always-valid 性质验证
- ✅ 清理 `__pycache__/` 和 `.coverage` 入库；补 `.gitignore`

---

## 文档

- [`docs/COMMON_PITFALLS.md`](docs/COMMON_PITFALLS.md) — A/B 测试常见错误清单
- [`ROADMAP.md`](ROADMAP.md) — 下一步计划（heterogeneous treatment effect、proportion 之外的指标族、ratio metrics）

## 许可

MIT

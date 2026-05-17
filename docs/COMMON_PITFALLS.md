# A/B 测试常见统计错误

A/B 测试看起来很简单（"算个 z 检验，看 p 值"），但有几个反复出现的陷阱
会让结果完全错。这份文档列出最常见的几个，以及本项目对应的修复。

## 1. Peeking（边采集边看 + 第一次显著就停）

### 问题
团队每天看仪表盘，看到 `p < 0.05` 就宣布"显著、上线"。如果实验周期内
偷看 20 次，真实假阳性率约从 5% 涨到 25%。Stanford 商学院 Walsh 论文
的模拟显示，5 次偷看就能把 alpha 拉到 14%。

### v1 哪里错了
`dashboard.py` v1 的"序贯分析"页面里：

```python
for i, p in enumerate(daily_p_values):
    if p < 0.05:
        st.success(f"第 {i+1} 天达到显著性，可以提前停止实验！")
        break
```

这是经典的 peeking。

### v2 的修复
`sequential.py` 提供 mSPRT 和 alpha-spending 两种"统计上正确"的偷看
方法，dashboard 改为调用它们。

## 2. 多重检验未校正

### 问题
做 20 个 metric 的 A/B 测试，每个都跑 `p < 0.05`，平均会有 1 个"显著"
是假阳性。如果团队挑出"显著的 metric" 上线，本质是在 cherry-picking。

### 怎么做对
- 频率学派：用 Bonferroni（保守）或 Benjamini-Hochberg FDR
- 贝叶斯：用先验把"全局 null"概率拉高
- 工程实务：提前指定 1~2 个 primary metric，guardrail metrics 用 FDR

本项目 `statistical_test.py` 的 `bonferroni_correction` 和
`fdr_correction` 已经实现。

## 3. Welch's t-test 但用了 equal-var 的均值差 SE

### v1 的 bug
`statistical_test.py:153-154` 进 `equal_var=False` 分支时：

```python
se = np.sqrt(np.var(sample_1, ddof=1)/n1 + np.var(sample_2, ddof=1)/n2)
df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
                                                          # ^ NameError
```

`var1` 和 `var2` 只在 `equal_var=True` 分支被定义。

### v2 的修复
把 `var1 = np.var(sample_1, ddof=1)`、`var2 = np.var(sample_2, ddof=1)`
提前到 if 之外。

## 4. SUTVA 违反（用户之间互相影响）

### 问题
社交、双边市场、共享库存的场景里，control 用户和 treatment 用户的行为
互相影响（spillover）。例如打车 app 提价实验：treatment 用户少叫车 →
control 用户更容易打到车 → control 体验变好 → 估计的"提价负面影响"被
低估。

### 怎么做对
- 按地域 / 时间窗 / 用户簇分流（cluster randomization）
- 设计 switchback 实验
- 用 synthetic control 比较"无实验区"

本项目暂未实现 cluster-level 推断（ROADMAP 中）。

## 5. 早停后还用未调整的 CI

### 问题
即使用了 mSPRT 决定何时停，传统的 95% CI（基于 fixed-n 假设）在数据相
关停止下不再是 95% 覆盖率。

### 本项目的处理
`MSPRT._evaluate()` 返回的 `always_valid_ci` 是用 mSPRT 的 likelihood
ratio 反推的 always-valid 区间，对任意停止时间都有 ≥ 1 - alpha 的覆盖。
注意它比 fixed-n CI 宽。

## 6. 相对效应 vs 绝对效应

### 问题
README 写"MDE = 10%" 时，是"基线转化率 10% → 11%（绝对提升 1pp / 相对
提升 10%）"，还是"基线 → 基线+10pp"？v1 的样本量计算函数用相对效应，但
有些团队习惯用绝对效应，混了就错一个量级。

### 本项目的约定
`ExperimentDesigner.calculate_sample_size_per_group(baseline_rate, mde)`
里 `mde` 是**相对**效应。绝对效应请用
`calculate_sample_size_continuous(effect_size, std_dev)`。

## 参考阅读

- Johari, Pekelis, Walsh (2015). [Always Valid Inference](https://arxiv.org/pdf/1512.04922).
- Deng, Xu, Kohavi, Walker (2013). [Improving the Sensitivity of Online
  Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).
- Kohavi, Tang, Xu (2020). *Trustworthy Online Controlled Experiments*. Cambridge.
- Auer, Cesa-Bianchi, Fischer (2002). Finite-time analysis of the
  multiarmed bandit problem.

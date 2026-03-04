# 🧪 A/B 测试分析平台

专业的 A/B 测试设计和分析平台，提供完整的实验设计、统计检验、结果可视化和报告生成功能。

## ✨ 核心功能

### 1. 实验设计
- **样本量计算**: 基于基线转化率、MDE、统计功效自动计算所需样本量
- **流量分配设计**: 支持多变体流量分配
- **统计功效分析**: 计算和可视化统计功效

### 2. 统计检验
- **频率学派方法**:
  - 两比例 Z 检验
  - 卡方检验
  - 独立样本 T 检验
- **贝叶斯方法**:
  - Beta-Binomial 后验分布
  - 可信区间估计
  - 后验概率计算

### 3. 多重检验校正
- **Bonferroni 校正**: 保守的族系错误率控制
- **FDR 校正**: Benjamini-Hochberg 方法，控制错误发现率

### 4. 序贯分析
- **早期停止规则**: 在达到显著性时提前终止实验
- **Alpha 花费函数**: O'Brien-Fleming、Pocock 边界
- **样本量重估**: 基于中期结果调整

### 5. 结果可视化
- 置信区间图
- 后验分布图
- 提升度分布图
- 累积结果图
- 序贯分析图

### 6. 实验报告
- 自动生成 Markdown 格式报告
- 包含频率学派和贝叶斯分析结果
- 可下载分享

## 🚀 快速开始

### 安装依赖

```bash
cd ab-testing-platform
pip install -r requirements.txt
```

### 运行应用

```bash
streamlit run dashboard.py
```

浏览器自动打开 `http://localhost:8501`

## 📖 使用指南

### 1. 实验设计

1. 输入基线转化率 (如 10%)
2. 设置最小可检测效应 MDE (如 10%)
3. 选择显著性水平 α (通常 0.05)
4. 选择统计功效 (通常 0.8)
5. 点击"计算样本量"

**示例**:
- 基线转化率：10%
- MDE: 10%
- α: 0.05
- Power: 0.8
- **结果**: 每组需要约 14,000 个样本

### 2. 数据导入

**支持格式**: CSV 文件

**必需列**:
- 分组列 (group): 标识对照组/实验组
- 指标列 (metric): 转化指标 (0/1 或连续值)

**可选列**:
- 日期列 (date): 用于时间序列分析

**示例数据格式**:
```csv
group,converted,date
control,0,2024-01-01
control,1,2024-01-01
treatment,1,2024-01-01
treatment,0,2024-01-01
```

### 3. 统计检验

选择检验方法:
- **两比例 Z 检验**: 适用于二元转化指标
- **贝叶斯 A/B 检验**: 提供后验概率和可信区间
- **综合检验**: 同时运行频率学派和贝叶斯方法

### 4. 结果解读

**频率学派**:
- P 值 < 0.05: 统计显著
- 置信区间不包含 0: 效应显著
- 效应量 (Cohen's h): >0.2 小效应，>0.5 中效应，>0.8 大效应

**贝叶斯**:
- P(实验组更优) > 95%: 强证据支持实验组
- 期望提升：预期提升百分比
- ROPE 概率：实际无差异的概率

### 5. 序贯分析

适用于:
- 长期实验 (7-30 天)
- 需要早期停止决策
- 监控每日结果

**早期停止规则**:
- P 值 < 0.05: 可以考虑提前停止
- 但需注意多重检验问题

## 📊 示例数据

平台提供内置示例数据用于测试:
- 对照组：10,000 样本，10% 转化率
- 实验组：10,000 样本，11% 转化率
- 真实提升：10%

点击"加载示例数据"即可使用。

## 🧪 单元测试

运行测试:

```bash
pytest tests/ -v --cov=.
```

测试覆盖:
- 样本量计算
- 统计检验
- 多重检验校正
- 可视化功能

## 📁 项目结构

```
ab-testing-platform/
├── dashboard.py              # 主界面
├── experiment_designer.py    # 实验设计模块
├── statistical_test.py       # 统计检验模块
├── result_visualizer.py      # 结果可视化模块
├── requirements.txt          # 依赖
├── README.md                 # 说明文档
├── tests/                    # 单元测试
│   ├── __init__.py
│   ├── test_experiment_designer.py
│   ├── test_statistical_test.py
│   └── test_visualizer.py
├── data/                     # 示例数据
│   └── example_data.csv
└── reports/                  # 生成的报告
    └── .gitkeep
```

## 🔬 统计方法详解

### 样本量计算

使用两比例检验的样本量公式:

$$n = \frac{(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{\beta}\sqrt{p_1(1-p_1) + p_2(1-p_2)})^2}{(p_2 - p_1)^2}$$

其中:
- $p_1$: 基线转化率
- $p_2$: 期望转化率
- $\bar{p} = (p_1 + p_2) / 2$
- $z_{\alpha/2}$: 标准正态分布的临界值
- $z_{\beta}$: 统计功效对应的 Z 值

### 贝叶斯 A/B 检验

使用 Beta-Binomial 共轭先验:

$$\theta | data \sim Beta(\alpha + successes, \beta + failures)$$

默认使用无信息先验 Beta(1, 1)。

### 多重检验校正

**Bonferroni**:
$$\alpha_{adjusted} = \alpha / m$$

**Benjamini-Hochberg FDR**:
按 p 值排序，找到最大的 k 使得 $p_{(k)} \leq \frac{k}{m}\alpha$

## ⚠️ 注意事项

1. **样本量不足**: 可能导致统计功效不足，无法检测到真实效应
2. **多重检验**: 多次查看结果会增加假阳性率，建议使用序贯分析
3. **辛普森悖论**: 确保实验组和对照组在其他维度上平衡
4. **新奇效应**: 用户可能因新鲜感而非真实改进产生行为变化
5. **长期效应**: 短期显著不代表长期有效，建议持续监控

## 📚 参考资料

- Kohavi, R., et al. (2020). "Trustworthy Online Controlled Experiments"
- Gelman, A., et al. (2020). "Bayesian Data Analysis"
- Scott, S. L. (2015). "A Modern Bayesian Look at the Multi-Armed Bandit"

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

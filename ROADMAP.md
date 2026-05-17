# Roadmap

## v2（本次提交，已完成）

- ✅ `sequential.py` — mSPRT 与 alpha-spending（O'Brien-Fleming / Pocock）
- ✅ `cuped.py` — CUPED 方差缩减
- ✅ `mab.py` — ε-greedy / UCB1 / Thompson sampling 与模拟器
- ✅ `report.py` — 自包含 HTML 报告
- ✅ `docs/COMMON_PITFALLS.md` — 常见统计错误清单
- ✅ 修 `statistical_test.py` Welch's t-test 的 NameError
- ✅ 修 `dashboard.py` 序贯分析页的 peeking 错误
- ✅ 32 个新测试（mSPRT 用 100 次蒙特卡洛验证 always-valid 性质）
- ✅ 清 `__pycache__/` `.coverage` 入库 + 补 `.gitignore`

## v3（计划）

### 指标族扩展
- [ ] **Ratio metrics**（如人均订单数 = 订单数/用户数）的 delta method
      标准误，常规 z 检验在这类指标上是错的
- [ ] **Count metrics** with Poisson / NegBinomial 模型
- [ ] **Survival metrics**（留存曲线）的 log-rank test

### 异质处理效应（HTE）
- [ ] T-learner / S-learner / X-learner 三种 meta-learner
- [ ] CATE 可信区间（uplift random forest）
- [ ] 分群展示模板

### 实验设计增强
- [ ] **Cluster-level randomization** 推断（spillover 场景）
- [ ] **Switchback** 实验的标准误（共享市场场景）
- [ ] Stratified sampling 与 post-stratification

### 工程
- [ ] FastAPI 后端 + 多实验仪表盘（取代 single-experiment Streamlit）
- [ ] 实验数据 schema（pydantic）+ 从 Snowflake / BigQuery 直读
- [ ] CI/CD（GitHub Actions 跑 pytest）

## 不会做的

- 「为统计学家做」一切：不要做"我们重新发明 R 的 power.prop.test"这种
  事；优先填生产环境实际缺的环节
- 闭源依赖（不引入需要付费 license 的统计包）

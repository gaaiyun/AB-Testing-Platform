"""
A/B 测试分析平台 - 主界面
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json

# 导入自定义模块
from experiment_designer import ExperimentDesigner, ExperimentConfig, get_sample_size_summary
from statistical_test import StatisticalTester, run_comprehensive_test
from result_visualizer import ResultVisualizer

# 页面配置
st.set_page_config(
    page_title="A/B 测试分析平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""
    
    # 标题
    st.markdown("<h1 class='main-header'>🧪 A/B 测试分析平台</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏导航
    st.sidebar.title("导航")
    page = st.sidebar.radio(
        "选择功能",
        ["实验设计", "数据导入", "统计检验", "结果可视化", "序贯分析", "实验报告"]
    )
    
    if page == "实验设计":
        experiment_design_page()
    elif page == "数据导入":
        data_import_page()
    elif page == "统计检验":
        statistical_test_page()
    elif page == "结果可视化":
        visualization_page()
    elif page == "序贯分析":
        sequential_analysis_page()
    elif page == "实验报告":
        report_page()


def experiment_design_page():
    """实验设计页面"""
    st.header("📋 实验设计")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("参数设置")
        
        baseline_rate = st.number_input(
            "基线转化率 (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        ) / 100
        
        mde = st.number_input(
            "最小可检测效应 (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        ) / 100
        
        alpha = st.number_input(
            "显著性水平 (α)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.01
        )
        
        power = st.number_input(
            "统计功效 (1-β)",
            min_value=0.5,
            max_value=0.99,
            value=0.8,
            step=0.05
        )
        
        allocation_ratio = st.number_input(
            "实验组分配比例",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1
        )
    
    with col2:
        st.subheader("样本量计算结果")
        
        if st.button("计算样本量", type="primary"):
            config = ExperimentConfig(
                baseline_rate=baseline_rate,
                mde=mde,
                alpha=alpha,
                power=power,
                allocation_ratio=allocation_ratio
            )
            
            summary = get_sample_size_summary(config)
            
            st.metric("每组样本量", f"{summary['sample_size_per_group']:,}")
            st.metric("总样本量", f"{summary['total_sample_size']:,}")
            st.metric("统计功效", f"{summary['achieved_power']:.2%}")
            
            st.info(f"""
            **解读**:
            - 检测到 {mde*100:.1f}% 的相对提升需要每组 **{summary['sample_size_per_group']:,}** 个样本
            - 在 α={alpha} 水平下，统计功效为 {summary['achieved_power']:.2%}
            - 建议实验持续时间：根据日均流量计算
            """)
    
    # 流量分配计算器
    st.markdown("---")
    st.subheader("流量分配设计")
    
    col1, col2 = st.columns(2)
    with col1:
        total_traffic = st.number_input("日均总流量", min_value=100, value=10000, step=100)
        n_variants = st.number_input("变体数量", min_value=2, max_value=10, value=2)
    
    with col2:
        if st.button("计算流量分配"):
            variants = {f"Variant {i}": 1/n_variants for i in range(n_variants)}
            allocation = ExperimentDesigner.design_traffic_split(total_traffic, variants)
            
            for name, n in allocation.items():
                st.metric(name, f"{n:,} ({n/total_traffic:.1%})")


def data_import_page():
    """数据导入页面"""
    st.header("📊 数据导入")
    
    # 上传文件
    uploaded_file = st.file_uploader("上传 CSV 文件", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("数据预览")
        st.dataframe(df.head())
        
        st.subheader("数据摘要")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", len(df))
        with col2:
            st.metric("列数", len(df.columns))
        with col3:
            st.metric("缺失值", df.isnull().sum().sum())
        
        # 自动识别列
        st.subheader("列映射")
        col_options = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox("分组列", col_options)
            metric_col = st.selectbox("指标列", col_options)
        with col2:
            date_col = st.selectbox("日期列 (可选)", [None] + col_options)
            variant_names = st.text_input("变体名称 (逗号分隔)", "control,treatment")
        
        if st.button("处理数据", type="primary"):
            st.session_state['imported_data'] = {
                'df': df,
                'group_col': group_col,
                'metric_col': metric_col,
                'date_col': date_col,
                'variant_names': variant_names.split(',')
            }
            st.success("数据处理完成！请前往'统计检验'页面进行分析")
    
    # 示例数据
    st.markdown("---")
    st.subheader("或使用示例数据")
    
    if st.button("加载示例数据"):
        np.random.seed(42)
        n_control = 10000
        n_treatment = 10000
        
        control_conv = np.random.binomial(1, 0.10, n_control)
        treatment_conv = np.random.binomial(1, 0.11, n_treatment)
        
        example_df = pd.DataFrame({
            'group': ['control'] * n_control + ['treatment'] * n_treatment,
            'converted': np.concatenate([control_conv, treatment_conv]),
            'date': pd.date_range('2024-01-01', periods=n_control+n_treatment, freq='H')[:n_control+n_treatment]
        })
        
        st.session_state['imported_data'] = {
            'df': example_df,
            'group_col': 'group',
            'metric_col': 'converted',
            'date_col': 'date',
            'variant_names': ['control', 'treatment']
        }
        
        st.success("示例数据已加载！")
        st.dataframe(example_df.head())


def statistical_test_page():
    """统计检验页面"""
    st.header("📈 统计检验")
    
    if 'imported_data' not in st.session_state:
        st.warning("请先在'数据导入'页面导入数据")
        return
    
    data = st.session_state['imported_data']
    df = data['df']
    group_col = data['group_col']
    metric_col = data['metric_col']
    variant_names = data['variant_names']
    
    # 汇总统计
    st.subheader("汇总统计")
    summary = df.groupby(group_col)[metric_col].agg(['sum', 'count', 'mean'])
    summary.columns = ['Successes', 'Total', 'Conversion Rate']
    st.dataframe(summary)
    
    # 检验方法选择
    st.subheader("检验方法")
    test_method = st.selectbox(
        "选择检验方法",
        ["两比例 Z 检验", "贝叶斯 A/B 检验", "综合检验"]
    )
    
    alpha = st.number_input("显著性水平", min_value=0.001, max_value=0.5, value=0.05, step=0.01)
    
    if st.button("运行检验", type="primary"):
        # 提取数据
        control_data = df[df[group_col] == variant_names[0]][metric_col]
        treatment_data = df[df[group_col] == variant_names[1]][metric_col]
        
        control_successes = control_data.sum()
        control_total = len(control_data)
        treatment_successes = treatment_data.sum()
        treatment_total = len(treatment_data)
        
        tester = StatisticalTester()
        
        if test_method == "两比例 Z 检验":
            result = tester.two_proportion_z_test(
                control_successes, control_total,
                treatment_successes, treatment_total,
                alpha
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Z 统计量", f"{result.statistic:.3f}")
            with col2:
                st.metric("P 值", f"{result.p_value:.4f}")
            with col3:
                st.metric("显著性", "是 ✅" if result.significant else "否 ❌")
            
            st.info(f"""
            **结果解读**:
            - 效应量 (Cohen's h): {result.effect_size:.3f}
            - 95% 置信区间: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]
            - 相对提升：{result.additional_info['relative_lift']:.2%}
            """)
        
        elif test_method == "贝叶斯 A/B 检验":
            result = tester.bayesian_ab_test(
                control_successes, control_total,
                treatment_successes, treatment_total
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("实验组更优概率", f"{result['prob_treatment_better']:.2%}")
            with col2:
                st.metric("期望提升", f"{result['expected_lift']:.2%}")
            with col3:
                st.metric("ROPE 概率", f"{result['rope_prob']:.2%}")
            
            st.info(f"""
            **贝叶斯解读**:
            - 后验均值 (对照组): {result['posterior_mean_2']:.4f}
            - 后验均值 (实验组): {result['posterior_mean_1']:.4f}
            - 95% 可信区间：[{result['credible_interval'][0]:.4f}, {result['credible_interval'][1]:.4f}]
            """)
        
        elif test_method == "综合检验":
            result = run_comprehensive_test(
                control_successes, control_total,
                treatment_successes, treatment_total,
                alpha
            )
            
            st.subheader("频率学派结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("P 值", f"{result['frequentist']['p_value']:.4f}")
            with col2:
                st.metric("显著性", "是 ✅" if result['frequentist']['significant'] else "否 ❌")
            with col3:
                st.metric("效应量", f"{result['frequentist']['effect_size']:.3f}")
            
            st.subheader("贝叶斯结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("实验组更优概率", f"{result['bayesian']['prob_treatment_better']:.2%}")
            with col2:
                st.metric("期望提升", f"{result['bayesian']['expected_lift']:.2%}")
            with col3:
                st.metric("可信区间", f"{result['bayesian']['credible_interval'][0]:.3f} ~ {result['bayesian']['credible_interval'][1]:.3f}")
            
            st.success(f"**推荐决策**: {result['recommendation']}")
        
        # 存储结果
        st.session_state['test_results'] = result


def visualization_page():
    """结果可视化页面"""
    st.header("📊 结果可视化")
    
    if 'test_results' not in st.session_state:
        st.warning("请先运行统计检验")
        return
    
    result = st.session_state['test_results']
    
    # 可视化类型选择
    viz_type = st.selectbox(
        "选择可视化类型",
        ["置信区间图", "后验分布图", "提升度分布图", "综合仪表板"]
    )
    
    if viz_type == "置信区间图" and isinstance(result, dict) and 'frequentist' in result:
        freq_result = result['frequentist']
        fig = ResultVisualizer.plot_confidence_intervals(
            control_rate=freq_result['additional_info']['p2'],
            control_ci=freq_result['confidence_interval'],
            treatment_rate=freq_result['additional_info']['p1'],
            treatment_ci=freq_result['confidence_interval']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "后验分布图" and isinstance(result, dict) and 'bayesian' in result:
        bayes_result = result['bayesian']
        if 'samples_1' in bayes_result:
            fig = ResultVisualizer.plot_posterior_distributions(
                bayes_result['samples_1'],
                bayes_result['samples_2'],
                "Treatment",
                "Control"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("后验样本不可用，请运行贝叶斯检验")
    
    elif viz_type == "提升度分布图" and isinstance(result, dict) and 'bayesian' in result:
        bayes_result = result['bayesian']
        if 'samples_1' in bayes_result:
            lift = (bayes_result['samples_1'] - bayes_result['samples_2']) / bayes_result['samples_2']
            fig = ResultVisualizer.plot_lift_distribution(lift)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("后验样本不可用")
    
    elif viz_type == "综合仪表板":
        st.info("综合仪表板功能开发中...")


def sequential_analysis_page():
    """序贯分析页面"""
    st.header("🔄 序贯分析")
    
    st.markdown("""
    **序贯分析**允许在实验进行过程中多次检验，而不需要增加假阳性率。
    
    支持方法:
    - **Alpha 花费函数**: O'Brien-Fleming, Pocock
    - **早期停止规则**: 显著性达到阈值时提前停止
    - **样本量重估**: 根据中期结果调整样本量
    """)
    
    # 模拟序贯分析
    st.subheader("序贯分析模拟")
    
    n_days = st.slider("实验天数", min_value=7, max_value=30, value=14)
    true_effect = st.slider("真实效应 (%)", min_value=-20.0, max_value=50.0, value=10.0) / 100
    baseline = st.slider("基线转化率 (%)", min_value=1.0, max_value=50.0, value=10.0) / 100
    
    if st.button("运行模拟", type="primary"):
        np.random.seed(42)
        
        daily_p_values = []
        daily_lifts = []
        cumulative_n = []
        
        control_cumulative = 0
        treatment_cumulative = 0
        control_successes = 0
        treatment_successes = 0
        
        for day in range(1, n_days + 1):
            # 每天新增样本
            daily_n = 1000
            control_new = np.random.binomial(1, baseline, daily_n)
            treatment_new = np.random.binomial(1, baseline * (1 + true_effect), daily_n)
            
            # 累积
            control_successes += control_new.sum()
            treatment_successes += treatment_new.sum()
            control_cumulative += daily_n
            treatment_cumulative += daily_n
            
            # 计算 p 值
            tester = StatisticalTester()
            result = tester.two_proportion_z_test(
                control_successes, control_cumulative,
                treatment_successes, treatment_cumulative
            )
            
            daily_p_values.append(result.p_value)
            daily_lifts.append((result.additional_info['p1'] - result.additional_info['p2']) / result.additional_info['p2'])
            cumulative_n.append(control_cumulative + treatment_cumulative)
        
        # 绘制序贯分析图
        fig = ResultVisualizer.plot_sequential_analysis(
            daily_p_values,
            alpha=0.05
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示结果
        st.subheader("模拟结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("最终 P 值", f"{daily_p_values[-1]:.4f}")
            st.metric("显著", "是 ✅" if daily_p_values[-1] < 0.05 else "否 ❌")
        with col2:
            st.metric("最终提升", f"{daily_lifts[-1]:.2%}")
            st.metric("总样本量", f"{cumulative_n[-1]:,}")
        
        # 检查早期停止
        for i, p in enumerate(daily_p_values):
            if p < 0.05:
                st.success(f"第 {i+1} 天达到显著性，可以提前停止实验！")
                break


def report_page():
    """实验报告页面"""
    st.header("📄 实验报告")
    
    if 'test_results' not in st.session_state:
        st.warning("请先运行统计检验")
        return
    
    result = st.session_state['test_results']
    
    # 报告生成
    st.subheader("实验报告生成")
    
    report_title = st.text_input("报告标题", "A/B 测试实验报告")
    experiment_name = st.text_input("实验名称", "Homepage CTA Button Test")
    experimenter = st.text_input("实验负责人", "")
    
    if st.button("生成报告", type="primary"):
        report = f"""
# {report_title}

## 实验信息
- **实验名称**: {experiment_name}
- **负责人**: {experimenter if experimenter else 'N/A'}
- **生成日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 实验结果摘要

### 频率学派分析
"""
        
        if isinstance(result, dict) and 'frequentist' in result:
            freq = result['frequentist']
            report += f"""
- **检验方法**: {freq['test_name']}
- **P 值**: {freq['p_value']:.4f}
- **统计显著性**: {'是 ✅' if freq['significant'] else '否 ❌'}
- **效应量**: {freq['effect_size']:.3f}
- **置信区间**: [{freq['confidence_interval'][0]:.4f}, {freq['confidence_interval'][1]:.4f}]
"""
        
        if isinstance(result, dict) and 'bayesian' in result:
            bayes = result['bayesian']
            report += f"""
### 贝叶斯分析
- **实验组更优概率**: {bayes['prob_treatment_better']:.2%}
- **期望提升**: {bayes['expected_lift']:.2%}
- **95% 可信区间**: [{bayes['credible_interval'][0]:.4f}, {bayes['credible_interval'][1]:.4f}]
"""
        
        if isinstance(result, dict) and 'recommendation' in result:
            report += f"""
## 推荐决策
**{result['recommendation']}**

## 后续建议
- 如果显著：考虑全量发布
- 如果不显著：继续收集数据或重新设计实验
- 监控长期指标，确保没有负面效应
"""
        
        st.session_state['report'] = report
        st.markdown(report)
        
        # 下载按钮
        st.download_button(
            label="下载报告 (Markdown)",
            data=report,
            file_name=f"ab_test_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()

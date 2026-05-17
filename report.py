"""
实验报告生成 — 输出可分享的自包含 HTML。

把 statistical_test / sequential / cuped 的结果聚合成一份完整的实验报告。
"""
from __future__ import annotations

import datetime as _dt
import html as _html
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>A/B 实验报告 — {title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
          color: #1f2937; max-width: 920px; margin: 32px auto; padding: 0 24px; line-height: 1.6; }}
  h1, h2, h3 {{ color: #0f172a; }}
  h1 {{ border-bottom: 3px solid #1e293b; padding-bottom: 8px; }}
  h2 {{ margin-top: 32px; border-left: 4px solid #2563eb; padding-left: 10px; }}
  .meta {{ color: #6b7280; font-size: 13px; }}
  .verdict {{ padding: 14px 18px; border-radius: 8px; margin: 16px 0; font-weight: 600; }}
  .verdict.ok {{ background: #ecfdf5; color: #065f46; border: 1px solid #6ee7b7; }}
  .verdict.no {{ background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }}
  .verdict.continue {{ background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
  th {{ background: #f9fafb; font-weight: 600; }}
  code, kbd, .num {{ font-family: "JetBrains Mono", "Menlo", "Consolas", monospace; }}
  .num {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }}
  .caveat {{ font-size: 13px; color: #6b7280; margin-top: 4px; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #e5e7eb;
            color: #9ca3af; font-size: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">生成时间：{ts} · 报告自包含，可直接发送或保存</p>

{verdict_block}

<h2>1. 实验信息</h2>
<table>{meta_rows}</table>

<h2>2. 频率学派检验</h2>
{freq_block}

<h2>3. 贝叶斯分析</h2>
{bayes_block}

{cuped_block}

{sequential_block}

<h2>方法学注记</h2>
<ul>
  <li>p-value &lt; α 仅意味着"在 H0 下观测到这样或更极端结果的概率小"，
      并不直接是"H1 为真"的概率。配合 Bayesian 后验概率一起读更稳。</li>
  <li>"早期停止"必须用 mSPRT 或 alpha-spending；不能裸跑 z 检验然后看到
      显著就停（peeking 会严重膨胀假阳性率）。</li>
  <li>若实验前有强相关 pre-experiment covariate，使用 CUPED 通常能减少
      30%~50% 样本量。</li>
</ul>

<footer>本报告由 AB-Testing-Platform 自动生成 · 报告内容不含外部依赖，可离线打开</footer>
</body>
</html>"""


def _verdict(verdict: str, prob_better: Optional[float] = None) -> str:
    if verdict == "ship":
        return f'<div class="verdict ok">✅ 建议上线（treatment 显著优于 control{f"，P(better)={prob_better:.1%}" if prob_better is not None else ""}）</div>'
    if verdict == "rollback":
        return f'<div class="verdict no">❌ 建议回滚（treatment 显著劣于或无效）</div>'
    return '<div class="verdict continue">⏳ 继续观察 / 样本不足以下结论</div>'


def _rows(d: Dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, float):
            v_str = f"{v:.4f}" if abs(v) < 1000 else f"{v:,.0f}"
        elif isinstance(v, tuple):
            v_str = f"[{v[0]:.4f}, {v[1]:.4f}]"
        else:
            v_str = _html.escape(str(v))
        rows.append(f"<tr><th>{_html.escape(str(k))}</th><td class='num'>{v_str}</td></tr>")
    return "\n".join(rows)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj.__dict__)


def render_report(
    title: str,
    meta: Dict[str, Any],
    frequentist: Optional[Dict[str, Any]] = None,
    bayesian: Optional[Dict[str, Any]] = None,
    cuped: Optional[Any] = None,
    sequential: Optional[Dict[str, Any]] = None,
) -> str:
    """渲染 HTML 字符串。所有部分可选；为 None 时该部分隐藏。"""
    verdict = "continue"
    prob_better = None
    if bayesian:
        prob_better = bayesian.get("prob_treatment_better")
        if prob_better is not None and prob_better >= 0.95:
            verdict = "ship"
        elif prob_better is not None and prob_better <= 0.05:
            verdict = "rollback"
    if frequentist and frequentist.get("significant") and verdict == "continue":
        diff = frequentist.get("additional_info", {}).get("difference", 0)
        verdict = "ship" if diff > 0 else "rollback"

    freq_block = "<p>未提供</p>"
    if frequentist:
        freq_block = "<table>" + _rows({
            "检验": frequentist.get("test_name", "—"),
            "统计量": frequentist.get("statistic"),
            "p-value": frequentist.get("p_value"),
            "显著": "是" if frequentist.get("significant") else "否",
            "效应量": frequentist.get("effect_size"),
            "95% 置信区间": frequentist.get("confidence_interval"),
        }) + "</table>"

    bayes_block = "<p>未提供</p>"
    if bayesian:
        bayes_block = "<table>" + _rows({
            "P(treatment > control)": bayesian.get("prob_treatment_better"),
            "期望相对提升": bayesian.get("expected_lift"),
            "可信区间": bayesian.get("credible_interval"),
            "control 后验均值": bayesian.get("posterior_mean_control"),
            "treatment 后验均值": bayesian.get("posterior_mean_treatment"),
        }) + "</table>"

    cuped_block = ""
    if cuped is not None:
        cd = _to_dict(cuped)
        cuped_block = "<h2>4. CUPED 方差缩减</h2><table>" + _rows({
            "θ": cd.get("theta"),
            "X-Y 相关系数": cd.get("correlation"),
            "方差缩减率": cd.get("variance_reduction"),
            "调整后 control 均值": cd.get("adjusted_mean_control"),
            "调整后 treatment 均值": cd.get("adjusted_mean_treatment"),
            "调整后均值差": cd.get("adjusted_diff"),
            "95% CI（调整后）": cd.get("ci_95"),
            "调整后 p-value": cd.get("p_value"),
        }) + "</table>"

    seq_block = ""
    if sequential:
        seq_block = "<h2>5. 序贯监控</h2><table>" + _rows(sequential) + "</table>"

    return _TEMPLATE.format(
        title=_html.escape(title),
        ts=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        verdict_block=_verdict(verdict, prob_better),
        meta_rows=_rows(meta),
        freq_block=freq_block,
        bayes_block=bayes_block,
        cuped_block=cuped_block,
        sequential_block=seq_block,
    )


def save_report(path: str | Path, **kwargs) -> Path:
    """渲染并写入文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_report(**kwargs), encoding="utf-8")
    return p

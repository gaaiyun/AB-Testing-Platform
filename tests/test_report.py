"""report.py 测试 — HTML 渲染结构与守门验证。"""
from pathlib import Path

import pytest

from cuped import cuped_two_sample
import numpy as np
from report import render_report, save_report


def test_render_minimal_report_includes_title():
    html = render_report(
        title="My Test",
        meta={"experiment_id": "exp_001"},
    )
    assert "<title>" in html
    assert "My Test" in html
    assert "exp_001" in html


def test_render_with_frequentist_and_bayes_marks_verdict():
    html = render_report(
        title="t",
        meta={"k": "v"},
        frequentist={
            "test_name": "Z",
            "statistic": 3.5,
            "p_value": 0.001,
            "significant": True,
            "effect_size": 0.05,
            "confidence_interval": (0.01, 0.09),
            "additional_info": {"difference": 0.05},
        },
        bayesian={
            "prob_treatment_better": 0.99,
            "expected_lift": 0.10,
            "credible_interval": (0.02, 0.18),
            "posterior_mean_control": 0.10,
            "posterior_mean_treatment": 0.11,
        },
    )
    assert "建议上线" in html


def test_render_low_prob_better_yields_rollback():
    html = render_report(
        title="t",
        meta={},
        bayesian={"prob_treatment_better": 0.02},
    )
    assert "建议回滚" in html


def test_render_no_inputs_returns_continue():
    html = render_report(title="t", meta={})
    assert "继续观察" in html


def test_render_cuped_block_included():
    rng = np.random.default_rng(0)
    n = 500
    x_c = rng.normal(10, 3, n)
    x_t = rng.normal(10, 3, n)
    y_c = x_c * 0.6 + rng.normal(0, 2, n)
    y_t = x_t * 0.6 + rng.normal(0.3, 2, n)
    r = cuped_two_sample(x_c, y_c, x_t, y_t)
    html = render_report(title="cuped-test", meta={}, cuped=r)
    assert "CUPED" in html
    assert "方差缩减率" in html


def test_save_report_writes_file(tmp_path: Path):
    out = save_report(
        tmp_path / "r.html",
        title="t",
        meta={"k": "v"},
    )
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

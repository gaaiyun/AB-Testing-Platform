"""
Pytest 配置文件
"""
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_data():
    """示例数据 fixture"""
    return {
        'control_successes': 1000,
        'control_total': 10000,
        'treatment_successes': 1200,
        'treatment_total': 10000
    }


@pytest.fixture
def sample_config():
    """示例实验配置 fixture"""
    return {
        'baseline_rate': 0.10,
        'mde': 0.10,
        'alpha': 0.05,
        'power': 0.8
    }

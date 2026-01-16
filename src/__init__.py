"""
VLM-VLA Agent 项目

模块说明：
    - config: 项目配置管理
    - model: VLM-VLA 模型架构
    - dataset: 数据加载和预处理
    - train: 训练脚本
    - inference: 推理脚本
    - utils: 工具函数
"""

__version__ = "0.1.0"

from .config import Config
from .model import VLM_ActionAgent, create_model
from .inference import VLMInference

__all__ = [
    'Config',
    'VLM_ActionAgent',
    'create_model',
    'VLMInference',
]


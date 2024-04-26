import os

import torch
from torch.nn import Module

from update_utils import path_util


def save_torch_model(model: Module, relative_path: str) -> None:
    """
    保存给定的 PyTorch 模型到指定的相对路径。

    该函数首先将相对路径转换为绝对路径，然后创建必要的目录，
    最后保存模型的状态字典（state_dict）。

    Args:
        model (Module): 要保存的 PyTorch 模型对象。
        relative_path (str): 模型应保存到的相对路径。

    Returns:
        None: 函数不返回任何值，但会打印模型保存的位置。
    """
    # 获取绝对路径
    absolute_path = path_util.get_absolute_path(relative_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    # 保存模型的状态字典
    torch.save(model.state_dict(), absolute_path)

    # 打印模型保存的位置
    print("Saved to:", absolute_path)

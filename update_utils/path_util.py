import os
from pathlib import Path


def get_project_root() -> Path:
    """
    获取项目根目录的绝对路径。

    此函数假设本文件位于项目根目录的某个子目录中。

    Returns:
        Path: 项目根目录的Path对象。
    """
    current_file = os.path.abspath(__file__)
    current_path = Path(current_file)
    root_path = current_path.parent.parent  # 根据项目结构调整
    return root_path


def get_absolute_path(relative_path: str) -> Path:
    """
    给定一个相对于项目根目录的相对路径，返回其绝对路径。

    Args:
        relative_path (str): 项目根目录的相对路径字符串。

    Returns:
        Path: 相对于项目根目录的绝对路径的Path对象。
    """
    root_path = get_project_root()
    absolute_path = root_path / relative_path
    return absolute_path


def convert_path_to_linux_style(path: str) -> str:
    """
    将Windows风格的路径字符串转换为Linux风格。

    Args:
        path (str): 需要转换的路径字符串。

    Returns:
        str: 转换后的Linux风格的路径字符串。
    """
    return str(path).replace("\\", "/")


# 使用示例
if __name__ == "__main__":
    print("项目根目录:", get_project_root())
    print("指定文件的绝对路径:", get_absolute_path("some/relative/path/to/file.txt"))

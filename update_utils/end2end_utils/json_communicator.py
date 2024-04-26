import json
from pathlib import Path

from update_utils import path_util


class JsonCommunicator:
    def __init__(self, file_path: str = './end2end/communicate/parameters.json'):
        """初始化方法，设置JSON文件的路径"""
        self.file_path: Path = path_util.get_absolute_path(file_path)

    def get_all_in_str(self):
        data = self._load_or_initialize_data()
        return str(data)

    def erase_all(self):
        with open(self.file_path, 'w') as file:
            json.dump({}, file)

    def get(self, key_path):
        """读取指定key路径的值"""
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            keys = key_path.split('.')
            for key in keys:
                data = data.get(key, None)
                if data is None:
                    return None
            return data
        except FileNotFoundError:
            print(f"File {self.file_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.file_path}.")
            return None

    def set(self, key_path, value):
        """写入或更新指定key路径的值"""
        data = self._load_or_initialize_data()
        keys = key_path.split('.')
        d = data
        for key in keys[:-1]:  # 遍历路径上的所有键，除了最后一个
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}  # 如果路径不存在或者不是字典，则创建一个新字典
            d = d[key]
        d[keys[-1]] = value  # 设置最终的键值
        self._write_data(data)

    def update(self, key_path, value):
        """如果存在，则更新指定key路径的值，否则不执行任何操作"""
        data = self._load_or_initialize_data()
        keys = key_path.split('.')
        d = data
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):  # 如果路径不存在，直接返回
                return
            d = d[key]
        if keys[-1] in d:  # 只有最终键存在时才更新
            d[keys[-1]] = value
            self._write_data(data)

    def _load_or_initialize_data(self):
        """尝试加载JSON文件，如果失败则返回空字典"""
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_data(self, data):
        """将数据写回JSON文件"""
        try:
            with open(self.file_path, 'w') as file:
                json.dump(data, file, indent=4)
        except IOError as e:
            print(f"Error writing to {self.file_path}: {e}")


def create_empty_json(file_path):
    """创建一个空的JSON文件"""
    try:
        with open(file_path, 'w') as file:
            json.dump({}, file)  # 写入一个空字典到文件
        print(f"Empty JSON file created at {file_path}")
    except IOError as e:
        print(f"Error creating JSON file at {file_path}: {e}")


if __name__ == "__main__":
    # path = './end2end/communicate/parameters.json'
    # abs_path = path_util.get_absolute_path(path)
    # create_empty_json(abs_path)

    # JsonCommunicator().set('learning_rate', 0.00002)
    # JsonCommunicator().set('epochs', 20)
    # JsonCommunicator().set('factor', "auto")
    print(JsonCommunicator().get_all_in_str())
    JsonCommunicator().erase_all()
    print(JsonCommunicator().get_all_in_str())

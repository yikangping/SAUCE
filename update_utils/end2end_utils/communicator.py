from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from update_utils import path_util

CURRENT_MODEL_PATH_TXT = './end2end/communicate/model_path.txt'
CURRENT_DATASET_PATH_TXT = './end2end/communicate/dataset_path.txt'
CURRENT_IS_DRIFT_TXT = './end2end/communicate/is_drift.txt'
CURRENT_SPLIT_INDICES_TXT = './end2end/communicate/split_indices.txt'
CURRENT_RANDOM_SEED_TXT = './end2end/communicate/random_seed.txt'


class FileCommunicator:
    def __init__(self, file_path: str):
        self.abs_file_path = path_util.get_absolute_path(file_path)

    def get(self):
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        return content

    def set(self, content: str):
        with open(self.abs_file_path, 'w') as file:
            file.write(content)


class RandomSeedCommunicator:
    def __init__(self, txt_path: str = CURRENT_RANDOM_SEED_TXT):
        self.abs_txt_path = path_util.get_absolute_path(txt_path)

    def get(self) -> int:
        with open(self.abs_txt_path, 'r') as file:
            content = file.read()
        print(f"RandomSeedCommunicator.get: {int(content)}")
        return int(content)

    def update(self):
        new_seed = self.get() + 1
        with open(self.abs_txt_path, 'w') as file:
            file.write(str(new_seed))

    def set(self, new_seed: int):
        with open(self.abs_txt_path, 'w') as file:
            file.write(str(new_seed))


class PathCommunicator(FileCommunicator):
    def __init__(self, file_path: str, prompt: str):
        super().__init__(file_path)
        self.prompt = prompt  # 新增属性来区分类型

    def get(self) -> Path:
        """
        读取txt获取当前路径
        """
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        print(f"GET-{self.prompt}-PATH={content}")
        abs_path = path_util.get_absolute_path(content)
        return abs_path

    def set(self, new_path: str):
        """
        将new_path写入txt用于记录当前路径
        """
        with open(self.abs_file_path, 'w') as file:
            file.write(new_path)
        print(f"SET-{self.prompt}-PATH={new_path}")


class ModelPathCommunicator(PathCommunicator):
    def __init__(self, txt_path: str = CURRENT_MODEL_PATH_TXT):
        super().__init__(file_path=txt_path, prompt="MODEL")


class DatasetPathCommunicator(PathCommunicator):
    def __init__(self, txt_path: str = CURRENT_DATASET_PATH_TXT):
        super().__init__(file_path=txt_path, prompt="DATASET")


class DriftCommunicator(FileCommunicator):
    def __init__(self, file_path: str = CURRENT_IS_DRIFT_TXT):
        super().__init__(file_path)

    def get(self) -> bool:
        """
        读取txt，若为"true"则返回True，否则返回False
        """
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        return content == 'true'

    def set(self, is_drift: bool):
        """
        将"true"或"false"写入txt
        """
        with open(self.abs_file_path, 'w') as file:
            content = 'true' if is_drift else 'false'
            file.write(content)


class CommaSplitArrayCommunicator(FileCommunicator):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def get(self) -> list:
        """
        读取txt文件，将字符串分割为数组
        """
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        content_list = content.split(',') if content else []
        return content_list

    def set(self, array: list):
        """
        将数组转换为字符串并写入txt文件
        """
        array_str = ','.join(map(str, array))
        with open(self.abs_file_path, 'w') as file:
            file.write(array_str)


class SplitIndicesCommunicator(CommaSplitArrayCommunicator):
    def __init__(self, txt_path: str = CURRENT_SPLIT_INDICES_TXT):
        super().__init__(txt_path)

    def get(self) -> List[int]:
        """
        Reads a txt file, splits the string into an array, and attempts to convert each element to an integer.
        Raises an error if conversion fails.
        """
        with open(self.abs_file_path, 'r') as file:
            content = file.read()

        if not content:
            return []

        try:
            content_list = [int(item) for item in content.split(',')]
        except ValueError:
            raise ValueError("Conversion to int failed for one or more elements in the list.")

        return content_list


if __name__ == '__main__':
    # 示例用法
    model_path_communicator = ModelPathCommunicator()
    dataset_path_communicator = DatasetPathCommunicator()

    original_model_path = model_path_communicator.get()
    model_path_communicator.set('./models/new-model-path.pt')
    new_model_path = model_path_communicator.get()

    original_dataset_path = dataset_path_communicator.get()
    dataset_path_communicator.set('./datasets/new-dataset-path.pt')
    new_dataset_path = dataset_path_communicator.get()

    print(original_model_path)
    print(new_model_path)
    print(original_dataset_path)
    print(new_dataset_path)

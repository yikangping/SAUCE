from pathlib import Path

import numpy as np
import pandas as pd

import Naru.common as common
from update_utils import arg_util
from update_utils.path_util import get_absolute_path


class DatasetLoaderUtils:
    @staticmethod
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        # 删除所有值均为缺失值的列
        df = df.dropna(axis=1, how="all")

        # 选择数据类型为 'object' 的列（通常是字符串）
        df_obj = df.select_dtypes(["object"])

        # 遍历这些列，尝试将其转换为更合适的数据类型（整数或浮点数）
        for col in df_obj.columns:
            # 尝试转换为整数
            try:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            except ValueError:
                # 如果转换整数失败，尝试转换为浮点数
                try:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                except ValueError:
                    # 如果还是失败，说明确实是字符串，去除首尾的空白字符
                    df[col] = df[col].str.strip()

        # 将空字符串替换为缺失值
        df.replace("", np.nan, inplace=True)

        # 删除含有任何缺失值的行
        df.dropna(inplace=True)

        return df

    @staticmethod
    def handle_permuted_dataset(
            permute: bool,
            df: pd.DataFrame,
            permuted_csv_file_path,
            dataset_name: str
            # update_size: int=None
    ):
        if permute:
            columns_to_sort = df.columns

            sorted_columns = pd.concat(
                [
                    df[col].sort_values(ignore_index=True).reset_index(drop=True)
                    for col in columns_to_sort
                ],
                axis=1,
                ignore_index=True,
            )
            sorted_columns.columns = df.columns
            update_sample = sorted_columns.sample(frac=0.2)
            data = pd.concat([df, update_sample])
            landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int32)
            save_path = get_absolute_path(permuted_csv_file_path)
            data.to_csv(save_path, sep=",", index=None)
            print(f"{str(save_path)} Saved")
        else:
            # update_sample = df.sample(frac=0.2)
            data = df
            lenth = int(len(data) * 5 / 6)
            landmarks = lenth + np.linspace(0, len(data) - lenth, 2, dtype=np.int32)

        print(
            "data size: {}, total size: {}, split index: {}".format(
                len(df), len(data), landmarks
            )
        )
        del df

        if permute:
            del sorted_columns
            del update_sample

        # print("census data size: {}".format(data.shape))
        return common.CsvTable(dataset_name, data, cols=data.columns), landmarks


class CsvDatasetLoader:
    DICT_FROM_DATASET_TO_COLS = {
        "census": [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
        ],
        "forest": [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
    }

    @staticmethod
    def load_csv_dataset(
            dataset_name: str,
            batch_num=None,
            finetune=False
    ):
        cols = CsvDatasetLoader.DICT_FROM_DATASET_TO_COLS.get(dataset_name, [])
        # 读取数据
        csv_file = f"./data/{dataset_name}/{dataset_name}.csv"
        csv_file = get_absolute_path(csv_file)
        df = pd.read_csv(csv_file, usecols=cols, sep=",")
        # 处理数据
        df = DatasetLoaderUtils.clean_df(df)

        if batch_num != None:
            landmarks = int(len(df) * 10 / 12) + np.linspace(
                0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
            )
            df = df.iloc[: landmarks[batch_num]]

        if finetune:
            landmarks = int(len(df) * 10 / 12) + np.linspace(
                0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
            )
            return common.CsvTable(dataset_name, df, cols), landmarks

        # landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        # df = df.iloc[:landmarks[5]]

        print(f"load_csv_dataset - {dataset_name}", df.shape)
        return common.CsvTable(dataset_name, df, cols)

    @staticmethod
    def load_permuted_csv_dataset(dataset_name: str, permute=True):
        raw_csv_path = f"./data/{dataset_name}/{dataset_name}.csv"
        permuted_csv_file_path = f"./data/{dataset_name}/permuted_dataset.csv"
        if permute:
            csv_file = raw_csv_path
        else:
            csv_file = permuted_csv_file_path
        csv_file = get_absolute_path(csv_file)
        cols = CsvDatasetLoader.DICT_FROM_DATASET_TO_COLS.get(dataset_name, [])

        # 读取数据
        df = pd.read_csv(csv_file, usecols=cols, sep=",")
        print(df.shape)

        # 处理数据
        df = DatasetLoaderUtils.clean_df(df)

        return DatasetLoaderUtils.handle_permuted_dataset(
            permute, df, permuted_csv_file_path, dataset_name
        )

    @staticmethod
    def LoadPartlyPermutedCensus(filename="census.csv", num_of_sorted_cols=1):
        csv_file = "../data/census/{}".format(filename)
        csv_file = get_absolute_path(csv_file)
        cols = CsvDatasetLoader.DICT_FROM_DATASET_TO_COLS.get("census", [])
        assert num_of_sorted_cols < len(cols)

        df = pd.read_csv(csv_file, usecols=cols, sep=",")
        print(df.shape)
        # 处理数据
        df = DatasetLoaderUtils.clean_df(df)

        if num_of_sorted_cols == 0:
            update_sample = df.sample(frac=0.2)
            landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int)

            data = pd.concat([df, update_sample])
            del df
            del update_sample
            data.to_csv("permuted_dataset.csv", sep=",", index=None)
            return common.CsvTable("census", df, cols=df.columns)

        columns_to_sort = [df.columns[i] for i in range(num_of_sorted_cols)]
        columns_not_sort = [
            df.columns[i] for i in range(num_of_sorted_cols, len(df.columns))
        ]

        sorted_columns = pd.concat(
            (
                    [
                        df[col].sort_values(ignore_index=True).reset_index(drop=True)
                        for col in columns_to_sort
                    ]
                    + [df[col] for col in columns_not_sort]
            ),
            axis=1,
            ignore_index=True,
        )
        sorted_columns.columns = df.columns

        update_sample = sorted_columns.sample(frac=0.2)
        landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int)

        data = pd.concat([df, update_sample])
        del df
        del update_sample

        data.to_csv("permuted_dataset.csv", sep=",", index=None)
        return common.CsvTable("census", data, cols=data.columns)


class NpyDatasetLoader:
    @staticmethod
    def _load_npy_as_df(abs_npy_file_path):
        data: np.ndarray = np.load(abs_npy_file_path, allow_pickle=True)

        # Calculate the number of columns in the data
        num_columns = data.shape[1]

        # Generate column names based on the number of columns
        column_names = [f'col-{i + 1}' for i in range(num_columns)]

        # Create the DataFrame with the generated column names
        df = pd.DataFrame(data, columns=column_names)
        return df, column_names

    @staticmethod
    def load_npy_dataset_from_path(path: Path):
        # 读取数据
        df, cols = NpyDatasetLoader._load_npy_as_df(path)

        # 处理数据
        df = DatasetLoaderUtils.clean_df(df)

        print("load_npy_dataset_from_path - df.shape =", df.shape)

        return common.CsvTable('default', df, cols)

    @staticmethod
    def load_permuted_npy_dataset(dataset_name: str, permute=True):
        # 读取数据
        if dataset_name=='census':
            npy_file_path = f"./data/{dataset_name}/end2end/census_int.npy"
        else:
            npy_file_path = f"./data/{dataset_name}/end2end/{dataset_name}.npy"
        permuted_csv_file_path = f"./data/{dataset_name}/permuted_dataset.csv"
        if permute:
            # 读取原始npy文件
            abs_file_path = get_absolute_path(npy_file_path)
            df, cols = NpyDatasetLoader._load_npy_as_df(abs_file_path)
        else:
            # 读取permute后的npy文件
            abs_file_path = get_absolute_path(npy_file_path)
            df, cols = NpyDatasetLoader._load_npy_as_df(abs_file_path)
            '''
            abs_file_path = get_absolute_path(permuted_csv_file_path)
            df = pd.read_csv(abs_file_path, sep=",")
            '''

        # 处理数据
        df = DatasetLoaderUtils.clean_df(df)

        return DatasetLoaderUtils.handle_permuted_dataset(
            permute, df, permuted_csv_file_path, dataset_name
        )


class DatasetLoader:
    @staticmethod
    def load_dataset(dataset: str):
        """
        Loads the dataset.

        Args:
            dataset (str): The dataset to be loaded.

        Returns:
            The loaded dataset
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset)

        if dataset in ["census", "forest"]:
            table = CsvDatasetLoader.load_csv_dataset(dataset_name=dataset)
        elif dataset in ["bjaq", "power"]:
            abs_path = get_absolute_path(f"./data/{dataset}/{dataset}.npy")
            table = NpyDatasetLoader.load_npy_dataset_from_path(path=abs_path)
        else:
            raise ValueError(f"Unknown dataset name \"{dataset}\"")

        return table

    @staticmethod
    def load_permuted_dataset(dataset: str, permute: bool = False):
        """
        Loads the permuted dataset.

        Args:
            dataset (str): The dataset to be loaded.
            permute (bool, optional): Whether to permute the dataset. Defaults to False.

        Returns:
            tuple: The loaded dataset and the split indices.
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset)

        '''
        if dataset in ["census", "forest"]:
            table, split_indices = CsvDatasetLoader.load_permuted_csv_dataset(dataset_name=dataset, permute=permute)
        elif dataset in ["bjaq", "power"]:
            table, split_indices = NpyDatasetLoader.load_permuted_npy_dataset(dataset_name=dataset, permute=permute)
        '''
        
        # 更新数据会保存到npy文件，因此统一从npy文件读取数据
        if dataset in ["census", "forest", "bjaq", "power"]:
            table, split_indices = NpyDatasetLoader.load_permuted_npy_dataset(dataset_name=dataset, permute=permute)
        else:
            raise ValueError(f"Unknown dataset name \"{dataset}\"")

        return table, split_indices

    @staticmethod
    def load_partly_permuted_dataset(dataset: str, num_of_sorted_cols: int):
        """
        Loads the partly permuted dataset.

        Args:
            dataset (str): The dataset to be loaded.
            num_of_sorted_cols (int): The number of sorted columns.

        Returns:
            The loaded dataset
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset)

        if dataset == "census":
            table = CsvDatasetLoader.LoadPartlyPermutedCensus(num_of_sorted_cols=num_of_sorted_cols)
        else:
            raise ValueError(f"Unknown dataset name \"{dataset}\"")

        return table


class DatasetConverter:
    @staticmethod
    def convert_csv_into_npy(dataset_name: str):
        """
        Converts the format of dataset file from csv to npy.

        Args:
            dataset_name (str): The dataset to be converted.
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset_name)

        # 读取数据
        csv_file_path = f"./data/{dataset_name}/{dataset_name}.csv"
        abs_csv_file_path = get_absolute_path(csv_file_path)
        df = pd.read_csv(abs_csv_file_path, sep=",")
        # 处理数据
        df = DatasetLoaderUtils.clean_df(df)

        print(f"load_csv_dataset - {dataset_name}", df.shape)

        # 保存数据为.npy格式
        npy_file_path = f"./data/{dataset_name}/{dataset_name}.npy"
        abs_npy_file_path = get_absolute_path(npy_file_path)
        np.save(abs_npy_file_path, df.values)

        print(abs_npy_file_path, "Saved")


if __name__ == "__main__":
    # dataset = "forest"
    dataset = "census"
    dir_path = f"./data/{dataset}"
    DatasetConverter.convert_csv_into_npy(dataset_name=dataset)
    pass

import argparse
from enum import Enum, auto
from typing import List


class ArgType(Enum):
    DATA_UPDATE = auto()
    DATASET = auto()
    DEBUG = auto()
    DRIFT_TEST = auto()
    END2END = auto()
    EVALUATION_TYPE = auto()
    MODEL_UPDATE = auto()


DICT_FROM_ARG_TO_ALLOWED_ARG_VALS = {
    ArgType.DATASET: ["census", "forest", "bjaq", "power"],
    ArgType.EVALUATION_TYPE: ["estimate", "drift"],
}


def add_common_arguments(parser: argparse.ArgumentParser, arg_types: List[ArgType]):
    """
    根据提供的枚举值列表向解析器添加参数。
    """
    for arg_type in arg_types:
        if arg_type == ArgType.DATA_UPDATE:
            parser.add_argument(
                "--data_update",
                type=str,
                choices=["permute-opt", "permute", "sample", "single", "value", "tupleskew", "valueskew"],
                help="数据更新方法:permute (DDUp), sample (FACE), permute (FACE), single (our)",
            )
            parser.add_argument(
                "--update_size", type=int, default=20000, help="数据更新规模:default=20000"
            )
        if arg_type == ArgType.DATASET:
            parser.add_argument(
                "--dataset",
                type=str,
                choices=["bjaq", "census", "forest", "power"],
                required=True,
                help="选择数据集:bjaq, census, forest, power",
            )
        if arg_type == ArgType.DEBUG:
            parser.add_argument(
                "--debug",
                action="store_true",  # 当指定 --debug 时，值为 True；否则为 False
                default=False,
                help="启用调试模式",
            )
        if arg_type == ArgType.DRIFT_TEST:
            parser.add_argument(
                "--drift_test",
                type=str,
                choices=["js", "ddup"],
                help="漂移测试方法:js (JS-divergence), ddup",
            )
        if arg_type == ArgType.END2END:
            parser.add_argument(
                "--end2end",
                action="store_true",  # 当指定 --debug 时，值为 True；否则为 False
                default=False,
                help="启用端到端实验",
            )
            parser.add_argument(
                "--query_seed",
                type=int,
                default=1234,
                help="查询生成的随机种子",
            )
        if arg_type == ArgType.EVALUATION_TYPE:
            parser.add_argument(
                "--eval_type",
                type=str,
                choices=["estimate", "drift"],
                required=True,
                help="选择评估类型:estimate, drift",
            )
        if arg_type == ArgType.MODEL_UPDATE:
            parser.add_argument(
                "--model_update",
                type=str,
                choices=["update", "adapt", "finetune"],
                help="模型更新方法:update (drift_test=ddup), adapt (drift_test=js), finetune (baseline)",
            )
            parser.add_argument(
                "--model",
                type=str,
                choices=["naru", "face", "transformer"],
                help="基础模型:naru, face",
            )
            



def validate_argument(arg_type: ArgType, arg_val: str):
    """
    Validates if the provided argument is allowed.

    Args:
        arg_type (ArgType): The type of the argument to be validated.
        arg_val (str): The value of the argument to be validated.

    Raises:
        ValueError: If the argument value is not in the allowed list.
    """
    if arg_val not in DICT_FROM_ARG_TO_ALLOWED_ARG_VALS.get(arg_type, []):
        raise ValueError(f'Validate Argument: UNKNOWN {arg_type.name}="{arg_val}"')
    print(f'Validate Argument: {arg_type.name}="{arg_val}" is valid.')


if __name__ == "__main__":
    # validate_argument(ArgType.DATASET, "census")
    # validate_argument(ArgType.EVALUATION_TYPE, "estimate")
    pass

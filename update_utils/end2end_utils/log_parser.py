import ast
import re
import sys

sys.path.append("./")
from pathlib import Path
from typing import List

import numpy as np

from update_utils import path_util, log_util

MODEL_NAME=['face', 'naru', 'transformer']
UPDATE_TYPE=['sample', 'permute-opt', 'single', 'value', 'tupleskew', 'valueskew']
DATASET_NAME=['bjaq', 'census', 'forest', 'power']

def parse_lines_with_keywords(src_path: Path, dst_path: Path, start_words: List[str]):
    with src_path.open("r") as src_file, dst_path.open("w") as dst_file:
        for line in src_file:
            if any(line.startswith(word) for word in start_words):
                dst_file.write(line)
                dst_file.write("\n")


def parse_experiment_records(
    src_dir: str,
    dst_dir: str,
    start_words: List[str] = None,  # 以此开头的行会被写入
    var_names: List[str] = None,  # 待求和的标量
    list_names: List[str] = None,  # 待统计的数组变量
):
    """
    整理src_dir下的所有txt实验记录，输出到dst_dir下

    使用：
        若需要重新生成：直接删除dst_dir，再次运行本函数
    """
    # 赋默认值
    if start_words is None:
        start_words = [
            "Input arguments",
            "JSON-passed parameters",
            "Experiment Summary",
            "Mean JS divergence",
            "WORKLOAD-FINISHED",
            "ReportEsts",
        ]
    if var_names is None:
        var_names = ["Model-update-time"]
    if list_names is None:
        list_names = ["ReportEsts"]

    src_dir_path = path_util.get_absolute_path(src_dir)
    dst_dir_path = path_util.get_absolute_path(dst_dir)

    # 确保目标文件夹存在
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    
    def err_dict_init():
        err_dict={}
        for dataset in DATASET_NAME:
            err_dict[dataset]={}
            for ut in UPDATE_TYPE:
                err_dict[dataset][ut]={}
                for model_name in MODEL_NAME:
                    err_dict[dataset][ut][model_name]=[]

        return err_dict
                
    errs=err_dict_init()
    err_labels=err_dict_init()

    # 处理源文件夹下的每个文件
    for src_file_path in src_dir_path.glob("*.txt"):
        dst_file_path = dst_dir_path / src_file_path.name

        # 跳过已处理的文件
        if dst_file_path.exists():
            continue
        
        group=dst_file_path.name.split("+")

        model=group[0]
        dataset=group[1]
        model_update_type=group[2]
        update_type=group[3]

        # 提取关键行并写入
        parse_lines_with_keywords(src_file_path, dst_file_path, start_words)

        # log_util.append_to_file(dst_file_path, "\n")

        # 统计求和变量
        for var_name in var_names:
            var_sum = sum_float_var_in_log(dst_file_path, var_name=var_name)
            log_util.append_to_file(
                dst_file_path, f"Sum of {var_name} = {var_sum:.4f}\n"
            )

        # 统计数组变量
        for list_name in list_names:
            concat_list, match_cnt = concat_list_var_in_log(
                dst_file_path, list_name=list_name
            )

            if match_cnt == 0:
                log_util.append_to_file(dst_file_path, f"Keyword {list_name} NOT found")
                continue

            # log_util.append_to_file(dst_file_path, f"Concatenated {list_name} = {concat_list}")
            if list_name == "ReportEsts":

                def generate_report_est_str(arr: list) -> str:
                    arr_max = np.max(arr)
                    quant99 = np.quantile(arr, 0.99)
                    quant95 = np.quantile(arr, 0.95)
                    arr_median = np.quantile(arr, 0.5)
                    arr_mean = np.mean(arr)
                    msg = (
                        f"max: {arr_max:.4f}\t"
                        f"99th: {quant99:.4f}\t"
                        f"95th: {quant95:.4f}\t"
                        f"median: {arr_median:.4f}\t"
                        f"mean: {arr_mean:.4f}\n"
                    )
                    return msg

                print(len(concat_list))
                tuple_len = int(len(concat_list) / match_cnt)
                first_query_errs = concat_list[:tuple_len]
                first_query_est_msg = (
                    "The 1st    ReportEsts -> "
                    + generate_report_est_str(first_query_errs)
                )
                log_util.append_to_file(dst_file_path, content=first_query_est_msg)

                after_query_errs = concat_list[tuple_len:]
                after_query_est_msg = (
                    "2nd to end ReportEsts -> "
                    + generate_report_est_str(after_query_errs)
                )
                log_util.append_to_file(dst_file_path, content=after_query_est_msg)

                # errs[dataset][update_type][model]+=[first_query_errs]
                errs[dataset][update_type][model]+=[after_query_errs]
                # err_labels[dataset][update_type][model].append(f"{model_update_type} 1st")
                err_labels[dataset][update_type][model].append(f"{model_update_type} 2nd-to-end")

                #画图
                # figure_name=dst_file_path.name.rstrip(".txt")
                # plot_box(first_query_errs, after_query_errs, plot_labels=["1st", "2nd-to-end"], file_name=figure_name)
        
    for dataset_name in DATASET_NAME:
        for ut in UPDATE_TYPE:
            for model_name in MODEL_NAME:
                if errs[dataset_name][ut][model_name]:
                    plot_name=f"{model_name}+{dataset_name}+{ut}"
                    plot_box(errs[dataset_name][ut][model_name], err_labels[dataset_name][ut][model_name], file_name=plot_name)



def sum_float_var_in_log(file_path: Path, var_name: str) -> float:
    total_sum = 0.0
    with file_path.open("r") as file:
        for line in file:
            if var_name in line:
                # Extract the value after the variable name and sum it up
                try:
                    value_str = line.split(var_name + ":")[1].strip()
                    total_sum += float(value_str)
                except (IndexError, ValueError):
                    # Handle cases where the line format is unexpected or the value is not a float
                    print(f"Warning: Could not parse line: {line.strip()}")
    return total_sum


def concat_list_var_in_log(file_path: Path, list_name: str):
    concatenated_list = []

    # Regular expression to find the desired list name and its contents
    pattern = rf"{re.escape(list_name)}: \[([^\]]*)\]"

    match_cnt = 0

    with file_path.open("r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the list and convert it to a Python list
                list_str = "[" + match.group(1) + "]"
                current_list = ast.literal_eval(list_str)

                # Concatenate to the main list
                concatenated_list.extend(current_list)
                match_cnt += 1

    return concatenated_list, match_cnt


def plot_box(arr: list, plot_labels, file_name: str):
    import matplotlib.pyplot as plt
    quarter=[np.quantile(errs, 0.9) for errs in arr]
    # quarter=[np.max(errs) for errs in arr]
    # y_max=np.max(quarter)
    y_max=5
    plt.ylim((0.5, y_max))
    plt.boxplot(arr,
                meanline=True,
                whis=2,
                meanprops={'color': 'blue', 'ls':'--', 'linewidth': 1.5},
                # showfliers=False,
                # flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10},
                labels=plot_labels)
    # plt.yticks(np.arange(1, y_max, 1))
    save_path=f'./end2end/figures/{file_name}.png'
    plt.savefig(save_path)
    plt.clf()
    # plt.show()
    

if __name__ == "__main__":
    src_dir: str = "./end2end/experiment-records"
    dst_dir: str = "./end2end/parsed-records"
    parse_experiment_records(src_dir=src_dir, dst_dir=dst_dir)

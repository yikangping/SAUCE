import sys
import yaml
from itertools import product
from pathlib import Path

sys.path.append("./")
from update_utils import path_util
from update_utils.end2end_utils.script_runner import PythonScriptRunner
from update_utils.end2end_utils.json_communicator import JsonCommunicator

ALLOWED_MODELS = ["naru", "face", "transformer"]


def filter_params_dict(args_dict: dict, params_dict: dict) -> dict:
    cur_model = args_dict["model"][0]
    unwanted_models = [model for model in ALLOWED_MODELS if model != cur_model]
    params_dict = params_dict.copy()  # To prevent modifying the original parameters

    # Delete parameters unrelated to the current model from params_dict
    for key in list(params_dict.keys()):
        # If the key starts with the name of other models, delete it
        if any(key.startswith(model) for model in unwanted_models):
            params_dict.pop(key)

    return params_dict


def run_multi_experiments(
        script_path: Path,
        args_dict: dict,
        params_dict: dict,
        check_experiments_order: bool = False
):
    # Iterate over each combination of arguments
    all_args_combinations = product(*args_dict.values())
    for combination in all_args_combinations:
        arg_dict = dict(zip(args_dict.keys(), combination))

        # Additional logic for model_update and dataset
        if arg_dict["model_update"] == "update":
            arg_dict["drift_test"] = "ddup"
        elif arg_dict["model_update"] == "adapt":
            arg_dict["drift_test"] = "js"

        if arg_dict["dataset"] == "census":
            arg_dict["update_size"] = 4000
        elif arg_dict["dataset"] == "forest":
            arg_dict["update_size"] = 25000
        elif arg_dict["dataset"] == "bjaq":
            arg_dict["update_size"] = 20000
        elif arg_dict["dataset"] == "power":
            arg_dict["update_size"] = 40000

        # Iterate over each combination of params
        print("Plan to run experiment(s) with arguments: \t", arg_dict)
        filtered_params: dict = filter_params_dict(args_dict=args_dict, params_dict=params_dict)  # filter irrelevant params
        all_params_combinations = product(*[[(k, v) for v in values] for k, values in filtered_params.items()])
        for params in all_params_combinations:
            # Set params
            for key, value in params:
                JsonCommunicator().set(key, value)
            # JsonCommunicator().print_all()

            # Run experiment
            print(f"> Run 1 sub-experiment with params: \t", params)
            if not check_experiments_order:
                runner = PythonScriptRunner(script_path=script_path, args=arg_dict)
                runner.run_script()

            # Clear params
            JsonCommunicator().erase_all()


if __name__ == "__main__":
    # Path for experiment parameters
    args_yaml_path = path_util.get_absolute_path("./end2end/args.yaml")
    # args_yaml_path = path_util.get_absolute_path("./end2end/args_v2.yaml")

    # Read experiment parameters
    with open(args_yaml_path, 'r', encoding='utf-8') as file:
        args_from_yaml = yaml.safe_load(file)  # Use safe_load to read the YAML file
    args_combination = args_from_yaml["args_combination"]
    params_combination = args_from_yaml["params_combination"]

    # Change params
    # args_combination["query_seed"]=[i for i in range(5, 20)]
    # params_combination["random_seed"]=[i for i in range(0, 5)]+[1226]
    
    '''
    # The following params would be passed to the experiment script as terminal arguments:
    args_combination = {
        "dataset": [
            # "power",
            "bjaq",
            # "forest",
            # "census"
        ],
        "model_update": [
            "update",
            # "adapt"
        ],
        "data_update": [
            # "single",
            # "permute-opt",
            # "sample",
            # "tupleskew",
            "valueskew",
        ],
        "model": [
            # "naru",
            # "face",
            "transformer",
        ],
        "num_workload": [
            # 50,
            3
        ],
        "query_seed": [14],
        # "query_seed":[i for i in range(0, 20)],
        # "query_seed": [7,8],
    }

    # The following params would be passed to the experiment script via json file:
    # Search the key name globally to find the usage of each param
    params_combination = {
        # >>> FACE <<<
        "face.learning_rate": [
            # 5e-06,
            # 5e-05,
            # 5e-04,
            1e-3,
            # 2e-3,
        ],
        "face.epochs": [
            # 30,
            # 40,
            # 50,
            1,
        ],
        "face.factor": [
            "auto",
            # "reverse",
            # 0.25,
            # 0.3,
            # 0.5,
            # 0.75,
            # 1,
        ],
        # >>> NARU <<<
        "naru.epochs": [
            # 30,
            1,
        ],
        # >>> TRANSFORMER <<<
        "transformer.epochs": [
            # 30,
            1,
        ]
    }
    '''

    # Run multiple sets of experiments
    # check_order_only = True  # If True: Only prints the order of experiments, does not run them
    check_order_only = False  # If False: Runs the experiments

    run_multi_experiments(
        script_path=path_util.get_absolute_path("./end2end/experiment.py"),  # Path to experiment script
        args_dict=args_combination,
        params_dict=params_combination,
        check_experiments_order=check_order_only
    )

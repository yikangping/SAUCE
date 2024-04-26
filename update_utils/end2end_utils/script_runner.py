import copy
from pathlib import Path
import subprocess
from typing import Dict, List, Any
from abc import ABC, abstractmethod


class BaseScriptRunner(ABC):
    """
    A base class for script runners.
    """

    def __init__(
            self,
            script_path: Path,
            args: Dict[str, Any],
            output_file_path: Path = None
    ):
        """
        Initialize the BaseScriptRunner with a script path and arguments.

        :param script_path: Path to the script to be run.
        :param args: A dictionary of argument names and values.
        :param output_file_path: Path to the log file.
        """
        self.script_path = script_path
        self.args = copy.deepcopy(args)
        self.log_file_path = output_file_path

    @abstractmethod
    def run_script(self):
        """
        Abstract method to execute the script. This method should be implemented in derived classes.
        """
        pass


class PythonScriptRunner(BaseScriptRunner):
    """
    A class to run a python script with given arguments using subprocess.
    """

    def _generate_python_script(self) -> List[str]:
        cmd = ["python", str(self.script_path)]
        for k, v in self.args.items():
            if v is None:
                cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}={v}")
        return cmd

    def run_script(self):
        """
        Execute the python script with the specified arguments using "subprocess.run".

        由于 "subprocess.run" 默认是阻塞性的，它会等待启动的进程结束后再继续执行。
        """
        # Construct the command with arguments
        cmd = self._generate_python_script()
        print(f"PythonScriptRunner.run_script -> Running command: {cmd}")

        # Execute the command
        if self.log_file_path:
            # redirect output to the file
            with open(self.log_file_path, 'a') as output_file:
                output_file.write(f"Going to run: {cmd}\n")
                output_file.flush()
                subprocess.run(cmd, stdout=output_file, stderr=output_file)
                # output_file.write("\n\n\n")
        else:
            subprocess.run(cmd)

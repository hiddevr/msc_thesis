import inspect
import functools
from tqdm.notebook import tqdm
import os
import ast
from multiprocessing import Manager, Lock
import sys

lock = Lock()


class ProgressManager:

    def __init__(self):
        step_counter = StepCounter()
        self.lock = lock
        self.total_functions = step_counter.count_decorated_functions_in_module('ppm_benchmark', 'nested_function_call')
        self.main_pb = None
        self.current_step = 0
        self.no_steps = None
        self.sub_loop = None

    def create_main_pb(self, no_datasets):
        self.no_steps = no_datasets * self.total_functions
        with self.lock:
            self.main_pb = tqdm(total=self.no_steps, desc="Total progress", position=0)
            self.main_pb.set_postfix_str(f"Step {self.current_step}/{self.no_steps}")

    def update_main_pb(self):
        with self.lock:
            if self.main_pb:
                self.main_pb.update(1)
                self.current_step += 1
                self.main_pb.set_postfix_str(f"Step {self.current_step}/{self.no_steps}")
                self.main_pb.refresh()

    def create_sub_loop(self, iterable, loop_title):
        with self.lock:
            sub_loop = tqdm(iterable, desc=loop_title, position=1)
        return sub_loop


class StepCounter:
    def count_decorated_functions_in_file(self, file_path, decorator_name):
        with open(file_path, "r") as file:
            tree = ast.parse(file.read(), filename=file_path)
        return self.count_decorated_functions_in_node(tree, decorator_name)

    def count_decorated_functions_in_node(self, node, decorator_name):
        count = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef) or isinstance(child, ast.AsyncFunctionDef):
                for decorator in child.decorator_list:
                    if (isinstance(decorator, ast.Name) and decorator.id == decorator_name) or \
                            (isinstance(decorator, ast.Attribute) and decorator.attr == decorator_name):
                        count += 1
                count += self.count_decorated_functions_in_node(child, decorator_name)
            else:
                count += self.count_decorated_functions_in_node(child, decorator_name)
        return count

    def count_decorated_functions_in_module(self, module_path, decorator_name):
        total_count = 0
        for root, _, files in os.walk(module_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    total_count += self.count_decorated_functions_in_file(file_path, decorator_name)
        return total_count


class FunctionTracker:
    def __init__(self):
        self.current_function = None
        self.loop_length = None
        self.tracker_stack = []

    def __enter__(self):
        self.current_function = inspect.stack()[1].function
        self.tracker_stack.append(self.current_function)
        #print(f"Entering function: {self.current_function}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #print(f"Exiting function: {self.current_function}")
        self.tracker_stack.pop()
        if self.tracker_stack:
            self.current_function = self.tracker_stack[-1]
        else:
            self.current_function = None
        self.loop_length = None

    def track_loop(self, iterable, loop_title):
        self.loop_length = len(iterable)
        sub_loop = pm.create_sub_loop(iterable, loop_title)
        return sub_loop

    def function_decorator(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result
        return wrapper

    def nested_function_call(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.tracker_stack.append(func.__name__)
            self.current_function = func.__name__
            #print(f"Entering nested function: {self.current_function}")
            try:
                result = func(*args, **kwargs)
            finally:
                pm.update_main_pb()
                #print(f"Exiting nested function: {self.current_function}")
                self.tracker_stack.pop()
                if self.tracker_stack:
                    self.current_function = self.tracker_stack[-1]
                else:
                    self.current_function = None
            return result
        return wrapper


global pm
global ft
ft = FunctionTracker()
pm = ProgressManager()

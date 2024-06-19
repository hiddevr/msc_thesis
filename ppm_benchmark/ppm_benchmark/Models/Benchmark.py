class TaskNotFoundError(Exception):
    pass


class Benchmark:

    def __init__(self, name, tasks):
        self.name = name
        self.tasks = tasks

        self.evaluations_dict = dict()

    def get_tasks(self):
        return [task.name for task in self.tasks]

    def load_task(self, task_name):
        for task in self.tasks:
            if task.name == task_name:
                return task
        raise TaskNotFoundError(f"Task with name '{task_name}' not found.")

    def evaluate(self, task, predictions):
        eval_dict = dict()
        print(f'Evaluation metrics for task {task.name}:')
        for metric, score in task.evaluate(predictions):
            print(f'\t {metric}: {score}')
            eval_dict[metric] = score
        self.evaluations_dict[task.name] = eval_dict

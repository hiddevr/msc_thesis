class Run:

    def __init__(self, task, run_id):
        self.task = task
        self.run_id = run_id
        self.start_time = None
        self.end_time = None
        self.training_time = None
        self.test_metrics = dict()
        self.epochs = dict()
        self.hyper_params = dict()

    def to_dict(self):
        return {
            'task': self.task.name,
            'run_id': self.run_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'training_time': self.training_time,
            'test_metrics': self.test_metrics,
            'epochs': self.epochs,
            'hyper_params': self.hyper_params
        }

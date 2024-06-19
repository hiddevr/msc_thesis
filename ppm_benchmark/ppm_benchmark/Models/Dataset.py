class Dataset:

    def __init__(self, name, dataset_normalizer, dataset_loader, data_path, is_remote, data_owner):
        self.name = name
        self.dataset_normalizer = dataset_normalizer
        self.dataset_loader = dataset_loader
        self.data_path = data_path
        self.is_remote = is_remote
        self.data_owner = data_owner

    def _load(self):
        if self.is_remote:
            print(f'Downloading {self.name} dataset by {self.data_owner} from {self.data_path}')
        data = self.dataset_loader.load_data(self.data_path)
        return data

    def normalize(self):
        raw_data = self._load()
        normalized_data = self.dataset_normalizer.normalize(raw_data)
        return normalized_data

    def create_task(self):
        pass

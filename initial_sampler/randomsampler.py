from .sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, dataset, args_input, args_task):
        super().__init__(dataset, args_input, args_task)

    def sample(self, n):
        # Just calling the initialize_labels function of the dataset
        return self.dataset.initialize_labels(n)

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Generating next batch: train_features, train_labels = next(iter(train_dataloader))
# Data shuffling
# Generating validation data loader by calling BaseDataLoader.split_validation()


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers=1):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        # Split Train & Valid
        self.sampler, self.valid_sampler = self._split_sampler(
            split=self.validation_split
        )

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # Samples elements randomly from a given list of indices, without replacement.
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

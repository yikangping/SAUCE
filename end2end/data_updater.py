import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

sys.path.append("./")
from update_utils import path_util
from update_utils.end2end_utils import communicator
from update_utils.end2end_utils.json_communicator import JsonCommunicator


class Sampler(ABC):
    def __init__(
            self, update_fraction: float = 0.2, update_size: int = None, random_seed=1234
    ):
        self.update_fraction = update_fraction
        self.update_size = update_size
        self.random_seed = random_seed

    def get_update_size(self, data: np.ndarray):
        """
        Returns the specified update_size if given; otherwise, calculates update_size based on update_fraction
        """
        if self.update_size and self.update_size < data.shape[0] * 0.2:
            return self.update_size
        data_size = data.shape[0]
        self.update_size = int(data_size * self.update_fraction)
        return self.update_size

    @abstractmethod
    def sample(self, data: np.ndarray):
        pass


class PermuteSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("Permute - START")
        n_rows, n_cols = data.shape
        samples = np.empty(shape=(self.update_size, n_cols))

        np.random.seed(self.random_seed)
        for i in range(self.update_size):
            if i % 100 == 0:
                print("Permute - {}/{}".format(i, self.update_size))
            idxs = np.random.choice(range(n_rows), n_cols, replace=False)
            for j, idx in enumerate(idxs):
                samples[i, j] = data[idx, j]

        print("Permute - END")
        return samples.astype(np.float32)


class PermuteOptimizedSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("Permute - START")
        n_rows, n_cols = data.shape
        samples = np.zeros((self.update_size, n_cols))

        np.random.seed(self.random_seed)
        for col in range(n_cols):
            samples[:, col] = np.random.choice(
                data[:, col], self.update_size, replace=True
            )

        print("Permute - END")
        return samples


class SingleSamplingSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("SingleSample - START")

        np.random.seed(self.random_seed)
        idx = np.random.randint(data.shape[0])
        sample_idx = [idx] * self.update_size
        sample = data[sample_idx]

        print("SingleSample - END")
        return sample


class SamplingSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("Sample - START")

        np.random.seed(self.random_seed)
        sample_idx = np.random.choice(
            range(data.shape[0]), size=self.update_size, replace=True
        )
        sample_idx = np.sort(sample_idx)
        sample = data[sample_idx]

        print("Sample - END")
        return sample


class ValueSamplingSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("ValueSample - START")

        np.random.seed(self.random_seed)
        row_idx = np.random.choice(range(data.shape[0]))
        column_idx = np.random.choice(range(data.shape[1]))
        value = data[row_idx][column_idx]
        sample_idx = np.random.choice(
            range(data.shape[0]), size=self.update_size, replace=True
        )
        sample_idx = np.sort(sample_idx)
        sample = data[sample_idx]
        sample[:, column_idx] = value

        # print(sample[:, column_idx])

        print("ValueSample - END")
        return sample


class TupleSkewSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("TupleSkewSample - START")

        # skew_ratio=1e-3
        skew_ratio = float(JsonCommunicator().get('data_updater.skew_ratio'))
        print(skew_ratio)
        n_rows, n_cols = data.shape

        if not self.random_seed == "auto":
            np.random.seed(int(self.random_seed))
        candidate_idx = np.random.choice(n_rows, size=round(skew_ratio * n_rows), replace=False)
        sample_idx = np.random.choice(candidate_idx, size=self.update_size, replace=True)
        sample_idx = np.sort(sample_idx)
        sample = data[sample_idx]

        # print(sample[:, column_idx])

        print("TupleSkewSample - END")
        return sample


class ValueSkewSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("ValueSkewSample - START")

        skew_ratio = float(JsonCommunicator().get('data_updater.skew_ratio'))
        print(skew_ratio)
        n_rows, n_cols = data.shape
        max_n_cols = n_cols // 3
        min_n_cols = 1

        if not self.random_seed == "auto":
            np.random.seed(int(self.random_seed))
        # update_n_cols=np.random.randint(min_n_cols, max_n_cols)
        # update_n_cols = 2
        update_n_cols = max_n_cols
        update_col_idx = np.random.choice(range(n_cols), size=update_n_cols, replace=False)
        # update_col_idx=[0,2]

        sample_idx = np.random.choice(n_rows, size=self.update_size, replace=True)
        sample_idx = np.sort(sample_idx)
        samples = data[sample_idx]
        for update_col in update_col_idx:
            candidate_values = np.random.choice(data[:, update_col], size=round(skew_ratio * n_rows), replace=False)
            samples[:, update_col] = np.random.choice(candidate_values, size=self.update_size, replace=True)

        # print(sample[:, column_idx])

        print("ValueSkewSample - END")
        return samples


def create_sampler(
        sampler_type: str,
        update_fraction: float = 0.2,
        update_size: int = None,
        random_seed=1234,
) -> Sampler:
    if sampler_type == "sample":
        return SamplingSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "permute":
        return PermuteSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "permute-opt":
        return PermuteOptimizedSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "single":
        return SingleSamplingSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "value":
        return ValueSamplingSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "tupleskew":
        return TupleSkewSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    if sampler_type == "valueskew":
        return ValueSamplingSampler(
            update_fraction=update_fraction,
            update_size=update_size,
            random_seed=random_seed,
        )
    raise ValueError(f"Unknown sampler type: {sampler_type}")


class DataUpdater:
    def __init__(self, data: np.ndarray, sampler: Sampler):
        self.sampler = sampler
        self.raw_data = data
        self.sampled_data = None
        self.updated_data = None

    def get_sampled_data(self):
        """Return sampled data."""
        return self.sampled_data

    def get_updated_data(self):
        """Return updated data."""
        return self.updated_data

    def update_data(self):
        """Use sampler to sample data and merge with the raw data."""
        self.sampled_data = self.sampler.sample(self.raw_data)
        self.updated_data = np.vstack((self.raw_data, self.sampled_data))

    def store_updated_data_to_file(self, output_path: Path):
        """Save the updated data to the specified output path."""
        np.save(output_path, self.updated_data)

    @staticmethod
    def update_dataset_from_file_to_file(
            data_update_method: str,
            raw_dataset_path: Path,
            updated_dataset_path: Path,
            update_fraction: float = None,
            update_size: int = None,
            random_seed=1234,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the current dataset from the original path, update it, and save the updated dataset to a new path."""
        # Load the current dataset from the original path
        raw_data = np.load(raw_dataset_path, allow_pickle=True)  # Original data

        # Update the data
        updater = DataUpdater(
            data=raw_data,
            sampler=create_sampler(
                sampler_type=data_update_method,
                update_fraction=update_fraction,
                update_size=update_size,
                random_seed=random_seed,
            ),
        )  # Create DataUpdater
        updater.update_data()  # Perform data update
        sampled_data = updater.get_sampled_data()  # New data added

        # Save the updated data to the new path
        updater.store_updated_data_to_file(output_path=updated_dataset_path)

        # Calculate and save landmarks
        original_data_end = len(raw_data)
        updated_data_end = original_data_end + len(sampled_data)
        landmarks = [original_data_end, updated_data_end]
        communicator.SplitIndicesCommunicator().set(landmarks)  # Save landmarks to file

        return raw_data, sampled_data


if __name__ == "__main__":
    # config
    dataset_name = "bjaq"
    assert dataset_name in ["census", "forest", "bjaq", "power"]
    raw_file_path = f"./data/{dataset_name}/{dataset_name}.npy"
    # raw_file_path = f"./data/{dataset_name}/end2end-{dataset_name}.npy"
    new_file_path = f"./data/{dataset_name}/end2end-{dataset_name}.npy"
    abs_raw_file_path = path_util.get_absolute_path(raw_file_path)
    abs_new_file_path = path_util.get_absolute_path(new_file_path)

    # load raw npy data
    raw_data = np.load(abs_raw_file_path)
    print(raw_data.shape)

    # sampler
    samplers = [
        # PermuteOptimizedSampler(update_fraction=0.2),
        # SingleSamplingSampler(update_fraction=0.2),
        # SamplingSampler(update_fraction=0.2),
        # ValueSamplingSampler(update_fraction=0.2),
        # TupleSkewSampler(update_fraction=0.2)
        ValueSamplingSampler(update_fraction=0.2)
    ]

    # updater
    for sampler in samplers:
        print("New sampler")
        updater = DataUpdater(raw_data, sampler=sampler)
        updater.update_data()
        updated_data = updater.get_updated_data()
        print(updated_data[:, 4])
        updater.store_updated_data_to_file(abs_new_file_path)

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os


class AnneDataset(Dataset):
    """Loader for preprocessed Anne data. The path should be a folder with npy files."""

    def __init__(self, path: str, device: str,
                 train_val_test_prop: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> None:
        """
        max_sequence_length acts as such:
        - If > 0, recordings are partitioned into random chunks of the given size. A random one is returned.
        - If = 0, no partitioning occurs.
        - If < 0, same as > 0, but all chunks of recordings are returned.
        """
        super().__init__()
        assert (sum(train_val_test_prop) == 1 and all(x >= 0 for x in train_val_test_prop))

        self.device = device
        self.files = self.find_data_files(path)
        self.num_labels = 2
        self.window_size, self.num_dims, self.num_dims_frequency, self.num_dims_scalar = self.check_data_shape()
        self.split_into_chunks = True

        # Split dataset
        train_prop, val_prop, test_prop = train_val_test_prop
        seed = 0
        if train_prop == 1:
            self.training, self.validation, self.testing = self.files, [], []
        else:
            self.training, self.testing = train_test_split(self.files,
                                                           train_size=train_prop,
                                                           random_state=seed)
            self.validation, self.testing = train_test_split(self.testing,
                                                             train_size=val_prop / (val_prop + test_prop),
                                                             random_state=seed)
        self.files = self.training

    def find_data_files(self, path: str) -> list[str]:
        """Find all npy files in a given file."""
        filenames = {}
        for root, dirs, files in os.walk(path):
            for filename in files:
                f = filename[:filename.find('_data')]
                if f not in filenames:
                    filenames[f] = [False, False, False, '']

                if filename.endswith('_data.npy'):
                    filenames[f][0] = True
                elif filename.endswith('_data_frequency.npy'):
                    filenames[f][1] = True
                elif filename.endswith('_data_scalar.npy'):
                    filenames[f][2] = True

                filenames[f][-1] = os.path.join(root, f)

        assert len(filenames) > 0, f"AnneDataset could not find any npy files in path '{path}'"
        files = []
        for f in filenames:
            assert filenames[f][0] and filenames[f][1] and filenames[f][2], f"Missing 1+ of feature/frequency/scalar data files for {f}"
            files.append(filenames[f][3])
        return files

    def check_data_shape(self) -> tuple[int, int, int, int]:
        """Check for consistency in shapes and feature count across data files"""
        window_size, dims, dims_frequency, dims_scalar = 0, 0, 0, 0
        for filename in self.files:
            s1 = np.load(f'{filename}_data.npy').shape
            s2 = np.load(f'{filename}_data_frequency.npy').shape
            s3 = np.load(f'{filename}_data_scalar.npy').shape

            assert len(s1) == 3,            f"[{filename}_data] Expected data of shape (B, D, N), got {s1}"
            assert len(s2) == 3,            f"[{filename}_data_frequency] Expected data of shape (B, D_freq, N_freq), got {s2}"
            assert len(s3) == 2,            f"[{filename}_data_scalar] Expected data of shape (B, D_scalar), got {s3}"
            assert s1[0] == s2[0] == s3[0], f"[{filename}] Inconsistent batch sizes across files, got {s1[0]}, {s2[0]}, {s3[0]}"
            assert s2[2] == s1[2] // 2,     f"[{filename}] Expected {s1[2] // 2} frequency bins, got {s2[1]}"

            if window_size == 0:        window_size = s1[2]
            else:                       assert s1[2] == window_size, f"[{filename}] Expected {window_size} window size, got {s1[2]}"
            if dims == 0:               dims = s1[1] - self.num_labels
            else:                       assert s1[1] - self.num_labels == dims, f"[{filename}_data] Inconsistent dimension sizes across files, got {s1[1]}, expected {dims}"
            if dims_frequency == 0:     dims_frequency = s2[1]
            else            :           assert s2[1] == dims_frequency, f"[{filename}_data_frequency] Inconsistent dimension sizes across files, got {s2[1]}, expected {dims_frequency}"
            if dims_scalar == 0:        dims_scalar = s3[1]
            else:                       assert s3[1] == dims_scalar, f"[{filename}_data_fscalar] Inconsistent dimension sizes across files, got {s3[1]}, expected {dims_scalar}"
        return window_size, dims, dims_frequency, dims_scalar

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filename = self.files[index] + '_data.npy'
        filename_frequency = self.files[index] + '_data_frequency.npy'
        filename_scalar = self.files[index] + '_data_scalar.npy'

        data = np.load(filename)
        data_frequency = np.load(filename_frequency)
        data_scalar = np.load(filename_scalar)

        assert np.all(~np.isnan(data)),             f"Detected NaN values in {filename}_data"
        assert np.all(~np.isnan(data_frequency)),   f"Detected NaN values in {filename}_data_frequency"
        assert np.all(~np.isnan(data_scalar)),      f"Detected NaN values in {filename}_data_scalar"

        if self.split_into_chunks:
            size = np.random.randint(data.shape[0] // 5, data.shape[0], size=1).item()
            i = np.random.randint(0, data.shape[0] - size, size=1).item()

            data = data[i:i + size]
            data_frequency = data_frequency[i:i + size]
            data_scalar = data_scalar[i:i + size]

        data = torch.from_numpy(data).float().to(self.device)
        data_frequency = torch.from_numpy(data_frequency).float().to(self.device)
        data_scalar = torch.from_numpy(data_scalar).float().to(self.device)

        # (N // K, D1, K)
        # (N // K, D2, K // 2)
        # (N // K, D3)
        # (N // K, num_labels)
        id = os.path.basename(self.files[index])
        return id, data[:, :-self.num_labels, :], data_frequency, data_scalar, torch.max(data[:, -self.num_labels:, :], dim=2).values

    def __len__(self) -> int:
        return len(self.files)


class AnneDataLoader(DataLoader):
    def __init__(self, dataset: AnneDataset, mode: str, split_into_chunks: bool = True,
                 **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        assert mode in {'training', 'validation', 'testing'}
        self.mode = mode
        self.split_into_chunks = split_into_chunks

    def __iter__(self):
        if self.mode == 'training':
            self.dataset.files = self.dataset.training
            self.dataset.split_into_chunks = self.split_into_chunks
        elif self.mode == 'validation':
            self.dataset.files = self.dataset.validation
            self.dataset.split_into_chunks = False
        else:
            self.dataset.files = self.dataset.testing
            self.dataset.split_into_chunks = False
        return super().__iter__()

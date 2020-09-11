import math
from typing import Tuple, Sequence, Optional
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def train_test_split(data: Sequence, train_size: float,
                     stratify: Optional[Sequence] = None) -> Tuple[Sequence, Sequence]:
    """
    Split data on two folds
    :param data: (Sequence) data for split
    :param train_size: (float) should be (0, 1)
    :param stratify: (Union[Sequence, None]) stratify folds
    :return: (Tuple[Sequence, Sequence]) train, test.
            Tuple[numpy.array, numpy.array] if stratify used,
    """
    if stratify is None:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_index = int(len(indices) * train_size)
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        return data[train_indices], data[test_indices]
    else:
        unique_values = np.unique(stratify)
        train_data = []
        test_data = []
        for u_value in unique_values:
            u_train, u_test = train_test_split(data[stratify == u_value], train_size, None)
            train_data.append(u_train)
            test_data.append(u_test)
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        return np.concatenate(train_data), np.concatenate(test_data)


class MyDataLoader:
    def __init__(self, data,
                 batch_size: int,
                 indices: Optional[Sequence] = None,
                 shuffle: bool = False,
                 transformer=None):
        """
        Create batches
        :param data: (Sequence)
        :param batch_size: (int)
        :param indices: (list or np.array) Default: None
        :param shuffle: (bool) Default: None
        :param transformer: (torchvision.transforms or other) transformer for augmentations. Default: None
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data = data
        if indices is None:
            self.data_len = len(data)
            self.indices = np.arange(self.data_len)
        else:
            self.data_len = len(indices)
            self.indices = np.array(indices)

        self.len_ = math.ceil(self.data_len / batch_size)

        if transformer is None:
            self.transformer = transforms.ToTensor()
        else:
            self.transformer = transformer

    def __len__(self) -> int:
        return self.len_

    def create_batch(self, indices: Sequence) -> Tuple[np.array, list]:
        X_batch = []
        y_batch = []
        for batch_index in indices:
            X, y = self.data[batch_index]
            X = self.transformer(X)
            X_batch.append(X)
            y_batch.append(y)
        if len(X_batch) > 1:
            X_batch = np.stack(X_batch)
        else:
            X_batch = np.unsqueeze(X_batch[0], 0)
        return X_batch, y_batch

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for n_batch in range(self.len_):
            start_index = n_batch * self.batch_size
            end_index = min(self.data_len, start_index + self.batch_size)
            batch_indices = self.indices[start_index: end_index]
            X_batch, y_batch = self.create_batch(batch_indices)
            yield X_batch, y_batch


class Dataset:
    def __init__(self, data_folder: Path, transform=np.array):
        """
        Load .jpg files from data folder
        :param data_folder: (pathlib.Path)
        :param transform: (transformer). Default: numpy.array
        """
        data_dir = Path(data_folder)
        self.files = list(data_dir.rglob('*.jpg'))
        self.len_ = len(self.files)
        if self.len_ == 0:
            raise FileNotFoundError(f'No files in {data_dir}')
        self.file_names = [path.name for path in self.files]
        self.transform = transform

    def __len__(self):
        return self.len_

    @staticmethod
    def load_sample(file):
        image = Image.open(file)
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = self.transform(x)
        name = self.file_names[index]
        return x, name
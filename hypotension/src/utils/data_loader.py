"""
    @ Author: DONGHEE KIM.
    @ Sungkyunkwan University and Hippo T&C all rights reserved.
"""
from .. import *
from . import *
from src.vitaldb_framework import vitaldb
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from typing import Callable, Tuple

import random
import glob
import torch
import h5py
import parmap
import gc


def data_load(data_path: str, attr: list, maxcases: int, interval: float) -> None:
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    _attributes = attr[2:]
    _ = vitaldb.load_cases(
        tnames=_attributes,
        path_for_save="{}/original".format(data_path),
        interval=interval,
        maxcases=maxcases
    )

# TODO: Need to be checked.
class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):

        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=True)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


def load_files(data_path: str, test_split_ratio: float, fixed=False) -> Tuple[list, list]:
    """
    Args:
        data_path: (str) Dataset path that holds hdf5 format.
        test_split_ratio: (float) Split ratio of train and test set.
        fixed: split deterministic or not

    Returns:

    """
    file_list = glob.glob(os.path.join(data_path, "*.hdf5"))

    if fixed:
        random.seed(42)

    random.shuffle(file_list)
    n_of_test = int(len(file_list) * test_split_ratio)
    test_file_list = file_list[:n_of_test]
    train_file_list = file_list[n_of_test:]

    return train_file_list, test_file_list


class BasicDataset(Dataset):
    def __init__(self, data_x, data_y):
        super(BasicDataset, self).__init__()
        self.x = data_x
        self.y = data_y

    def __getitem__(self, item):
        x = np.array(self.x[item], dtype=np.float64)
        y = np.array(self.y[item], dtype=np.int64)
        return x, y

    def __len__(self):
        return len(self.y)


class HDF5_VitalDataset(Dataset):
    def __init__(self, file_lists: list):
        super().__init__()
        self.x, self.y = self._load_from_hdf5(file_lists)

    def __getitem__(self, index):
        x = np.array(self.x[index], dtype=np.float64)
        y = np.array(self.y[index], dtype=np.int64)
        return x, y

    def __len__(self) -> int:
        """
        Returns:
            (int) length of data X
        """
        return len(self.x)

    def get_labels(self) -> list:
        """
        Returns:
            (list) List form of target Y.
        """
        return list(self.y)

    def get_shape(self) -> tuple:
        return self.x.shape, self.y.shape

    def label_counts(self) -> dict:
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))

    def _load_from_hdf5(self, file_lists: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            file_lists: (list) List of files to train or test.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Concatenated data from all data path.
        """
        result_x = []
        result_y = []

        result = parmap.map(read_HDF5, file_lists, pm_pbar=True, pm_processes=os.cpu_count())

        pbar = tqdm(total=len(result), desc='Making numpy array', mininterval=0.01)
        while result:
            res = result.pop()
            if len(res[0]) <= 0 or len(res[1]) <= 0:
                # Pass if there is no data at all.
                pbar.update(1)
                continue
            result_x.append(res[0])
            result_y.append(res[1])
            del res
            gc.collect()
            pbar.update(1)

        result_x = np.concatenate(result_x, axis=0)
        result_y = np.concatenate(result_y, axis=0)

        return result_x, result_y

    def set_valid(self) -> Dataset:
        result_0 = np.where(self.y == 0)[0]
        result_1 = np.where(self.y == 1)[0]

        sample_0 = np.random.choice(result_0, int(len(result_0)*0.1), replace=True)
        sample_1 = np.random.choice(result_1, int(len(result_1)*0.1), replace=True)

        indices = np.append(sample_0, sample_1, axis=0)
        valid_x = self.x[indices]
        valid_y = self.y[indices]

        self.x = np.delete(self.x, indices, axis=0)
        self.y = np.delete(self.y, indices, axis=0)

        valid_set = BasicDataset(valid_x, valid_y)
        return valid_set


def read_HDF5(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read HDF5 file.
    Args:
        file: (str) File path

    Returns:
        Tuple[np.ndarray, np.ndarray]: numpy array of data_x and data_y
    """
    f = h5py.File(file, 'r')
    data_x = np.array(f.get('x'), dtype=np.float16)
    data_y = np.array(f.get('y'), dtype=np.int64)
    f.close()
    return data_x, data_y

from typing import Tuple
import numpy as np
from tslearn.datasets import UCR_UEA_datasets
import torch
from torch.utils.data import Dataset, DataLoader
from enchanter.utils.datasets import TimeSeriesLabeledDataset

__all__ = ["UEADataset", "TSULUEADataset", "fetch_loader", "fetch_dataset"]


def fetch_dataset(name: str):
    loader = UCR_UEA_datasets()
    x_train, y_train, x_test, y_test = loader.load_dataset(name)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    def projector(targets):
        if name in {"PEMS-SF", "Libras"}:
            targets = targets.astype(float).astype(int) - 1
        elif name in {"UWaveGestureLibraryAll"}:
            targets = targets - 1
        else:
            project = {k: i for i, k in enumerate(np.unique(targets))}
            targets = np.array(list(map(lambda n: project[n], targets)))

        return targets

    y_train = projector(y_train).astype(np.int64)
    y_test = projector(y_test).astype(np.int64)

    return (x_train, y_train), (x_test, y_test)


class UEADataset(Dataset):
    def __init__(self, train=True, name: str = "RacketSports"):
        loader = UCR_UEA_datasets()
        projector = None
        if train:
            data, targets, _, _ = loader.load_dataset(name)
        else:
            _, _, data, targets = loader.load_dataset(name)

        if name in {"PEMS-SF", "Libras"}:
            targets = targets.astype(float).astype(int) - 1
        else:
            projector = {k: i for i, k in enumerate(np.unique(targets))}
            targets = np.array(list(map(lambda n: projector[n], targets)))

        self.data = data.transpose(0, 2, 1).astype(np.float32)
        self.targets = targets.astype(int)
        self.n_features = self.data.shape[1]
        self.n_targets = len(np.unique(self.targets))
        if projector is not None:
            self.projector = {k: v for v, k in projector.items()}
        else:
            self.projector = projector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = self.targets[item]

        return data, target


class TSULUEADataset(TimeSeriesLabeledDataset):
    def __init__(self, train, name):
        loader = UCR_UEA_datasets()
        projector = None
        if train:
            data, targets, _, _ = loader.load_dataset(name)
        else:
            _, _, data, targets = loader.load_dataset(name)

        if name in {"PEMS-SF", "Libras"}:
            targets = targets.astype(float).astype(int) - 1
        else:
            projector = {k: i for i, k in enumerate(np.unique(targets))}
            targets = np.array(list(map(lambda n: projector[n], targets)))

        data = torch.tensor(data.transpose(0, 2, 1).astype(np.float32))
        targets = torch.tensor(targets.astype(int))
        super(TSULUEADataset, self).__init__(data, targets)

        self.n_features = data.shape[1]
        self.n_targets = len(np.unique(targets))

        self.n_targets = len(np.unique(self.target))
        if projector is not None:
            self.projector = {k: v for v, k in projector.items()}
        else:
            self.projector = projector

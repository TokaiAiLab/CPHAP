import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from torch.utils.data import Dataset


__all__ = ["UEADataset"]


class UEADataset(Dataset):
    def __init__(self, train=True, name: str = "RacketSports"):
        loader = UCR_UEA_datasets()
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        target = self.targets[item]

        return data, target

from comet_ml import Experiment
from typing import Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.style
from quicksom.som import SOM

from torchscan import crawl_module
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA
from enchanter.engine.modules import fix_seed, get_dataset
from enchanter.tasks import ClassificationRunner

from cphap.functions import find_t, han, generate_indices_for_hap
from cphap.utils import fetch_dataset


fix_seed(0)
matplotlib.style.use("seaborn")


class CNN(nn.Module):
    def __init__(self, in_features, mid_features, n_class, depth: int = 1):
        super(CNN, self).__init__()
        self.conv = nn.ModuleList([nn.Conv1d(in_features, mid_features, kernel_size=3)])
        self.depth = depth
        tmp = [nn.Conv1d(mid_features, mid_features, kernel_size=3) for _ in range(depth - 1)]
        self.conv.extend(tmp)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(mid_features, n_class)

    def forward(self, x):
        batch = x.shape[0]

        for layer in self.conv:
            x = layer(x)

        out = self.pool(x)
        out = out.reshape(batch, -1)
        return self.fc(out)


def train_nn(dataset: str, batch_size: int, depth: int, epochs: int) -> Tuple[CNN, Tuple[np.ndarray, np.ndarray]]:
    experiment = Experiment(project_name="cphap", auto_output_logging=False)
    experiment.add_tag(dataset)
    experiment.add_tag("NN-depth-{}".format(depth))
    (x_train, y_train), (x_test, y_test) = fetch_dataset(dataset)
    scaler = TimeSeriesScalerMeanVariance()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.transpose((0, 2, 1)).astype(np.float32)
    x_test = x_test.transpose((0, 2, 1)).astype(np.float32)

    n_features = x_train.shape[1]
    n_targets = len(np.unique(y_train))

    train_ds = get_dataset(x_train, y_train)
    test_ds = get_dataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CNN(n_features, 32, n_targets, depth=depth)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    runner = ClassificationRunner(
        model, optimizer, criterion, experiment
    )
    runner.add_loader("train", train_loader)
    runner.add_loader("test", test_loader)

    runner.train_config(epochs=epochs)
    runner.run()
    runner.quite()

    return runner.model.eval(), (x_train, x_test)


def calculate_rf(model: CNN, dummy_in: Tuple[int, int]) -> List[int]:
    rf = []
    module_info = crawl_module(model, dummy_in)
    info = module_info["layers"][1:model.depth+1]

    for i in info:
        rf.append(int(i["rf"]))

    return rf


def calculate_thresholds(model: CNN, data: torch.Tensor) -> List[torch.Tensor]:
    results = []
    for layer in model.conv:
        data = layer(data)
        results.append(find_t(data))

    return results


def hap_core(
    hap_lists: List[List],
    model: CNN,
    x: torch.Tensor,
    j: int,
    k: int,
    thresholds: List[torch.Tensor],
    receptive_field
):
    data = x.unsqueeze(0)

    for count, layer in enumerate(model.conv):
        data = layer(data)
        if count == j:
            break

    han_jk, max_size = han(data, channel=k, thresholds=thresholds[j])
    rf_j = receptive_field[j]
    tmp = generate_indices_for_hap(han_jk, rf_j, max_size)
    if len(tmp) != 0:
        hap_lists[j].append(x[:, tmp])

    return hap_lists, tmp


def run_hap():
    """
    j == layer
    hap_list[j].shape == [in_channels, sub_seq, rf]

    """
    batch_size = 32
    depth = 2
    dataset_name = "RacketSports"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (x_train, x_test) = train_nn(dataset_name, batch_size, depth, epochs=100)
    x_train = torch.tensor(x_train, device=device)
    x_test = torch.tensor(x_test, device=device)
    thresholds = calculate_thresholds(model, torch.cat([x_train, x_test], dim=0))
    dummy: Tuple[int, int] = (x_train.shape[1], x_train.shape[2])
    rf = calculate_rf(model, dummy)

    hap_lists = [list() for _ in range(depth)]

    for x in x_test:
        for j in range(depth):
            for k in range(model.conv[j].out_channels):
                hap_lists, _ = hap_core(hap_lists, model, x, j, k, thresholds, receptive_field=rf)

    for i in range(depth):
        hap_lists[i] = torch.cat(hap_lists[i], 1)

    return hap_lists, (x_train, x_test), model


def train_som(data, map_size: Tuple[int, int], epochs):
    """

    Args:
        data: [N, sub_seq_len]
        map_size: ex, (8, 8)
        epochs:

    Returns:

    """
    device = data.device
    data = PCA(n_components=0.95).fit_transform(data.cpu().numpy()).astype(np.float32)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    columns = map_size[0]
    rows = map_size[1]
    som = SOM(columns, rows, data.shape[-1], niter=epochs, device=device)
    som.fit(data, print_each=epochs * 100)

    return som


def predict_cluster(trained_som: SOM, data: torch.Tensor) -> torch.Tensor:
    c = trained_som.predict(data)[0]
    return c


def main(data_idx: int, layer: int, in_channel, target_channel: int, epochs: int):
    cphaps = []
    uncertainties = []
    # hap_listはx_testに対するもの
    hap_lists, (x_train, x_test), model = run_hap()
    som = train_som(hap_lists[layer][in_channel], (8, 8), epochs)
    p = [list() for _ in range(model.depth)]
    thresholds = calculate_thresholds(model, torch.cat([x_train, x_test], dim=0))
    rf = calculate_rf(model, (x_train.shape[1], x_train.shape[2]))
    p, indices = hap_core(p, model, x_test[data_idx], layer, target_channel, thresholds, rf)

    p = p[layer][0][in_channel]
    predicts = predict_cluster(som, p)

    for predict in predicts:
        try:
            for_mean_variance = torch.stack(
                [p_ for p_ in hap_lists[layer][in_channel] if (predict == predict_cluster(som, p_)).all()]
            )
        except RuntimeError:
            pass
        else:
            cphaps.append(for_mean_variance.mean(0))
            uncertainties.append(for_mean_variance.var(0))

    plt.figure(figsize=(10, 5))
    plt.plot(x_test[data_idx][in_channel].cpu().numpy())
    if len(cphaps) != 0:
        for i in range(len(cphaps)):
            plt.fill_between(
                indices[i],
                (cphaps[i] - uncertainties[i]).cpu().numpy(),
                (cphaps[i] + uncertainties[i]).cpu().numpy(),
                alpha=0.3,
                color="g"
            )

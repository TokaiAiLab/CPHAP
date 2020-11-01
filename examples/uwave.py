from comet_ml import Experiment
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.style

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from enchanter.engine.modules import fix_seed, get_dataset
from enchanter.tasks import ClassificationRunner

from cphap.utils import fetch_dataset
from cphap.core import CNN, calculate_rf, calculate_thresholds, hap_core, train_som, predict_cluster


fix_seed(0)
matplotlib.style.use("seaborn")


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


def run_hap():
    """
    j == layer
    hap_list[j].shape == [in_channels, sub_seq, rf]

    """
    batch_size = 32
    depth = 2
    dataset_name = "UWaveGestureLibraryAll"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (x_train, x_test) = train_nn(dataset_name, batch_size, depth, epochs=100)
    x_train = torch.tensor(x_train, device=device)
    x_test = torch.tensor(x_test, device=device)
    thresholds = calculate_thresholds(model, torch.cat([x_train, x_test], dim=0))
    dummy: Tuple[int, int] = (x_train.shape[1], x_train.shape[2])
    rf = calculate_rf(model, dummy)

    hap_lists = [list() for _ in range(depth)]

    for x in tqdm(x_test):
        for j in range(depth):
            for k in range(model.conv[j].out_channels):
                hap_lists, _ = hap_core(hap_lists, model, x, j, k, thresholds, receptive_field=rf)

    for i in range(depth):
        hap_lists[i] = torch.cat(hap_lists[i], 1)

    return hap_lists, (x_train, x_test), model



class CPHAPFrontend:
    def __init__(self, som_map_size: Tuple[int, int] = (8, 8)):
        self.x_train = None
        self.x_test = None
        self.model = None
        self.hap_lists = None
        self.som = None
        self.som_map_size = som_map_size
        self.p = None
        self.predicts = None
        self.indices = None
        self.target_layer = None
        self.target_in_channel = None
        self.data_idx = None

    def train_nn(self):
        self.hap_lists, (self.x_train, self.x_test), self.model = run_hap()
        self.reset_p()

    def train_som(self, layer: int, epoch: int):
        self.target_layer = layer
        self.som = train_som(self.hap_lists[layer][0], self.som_map_size, epoch)

    def compute_p(self, data_idx, target_channel):
        self.data_idx = data_idx
        layer = self.target_layer
        thresholds = calculate_thresholds(self.model, torch.cat([self.x_train, self.x_test], dim=0))
        rf = calculate_rf(self.model, (self.x_train.shape[1], self.x_train.shape[2]))
        p, indices = hap_core(self.p, self.model, self.x_test[data_idx], layer, target_channel, thresholds, rf)
        if len(p[layer]) != 0:
            p = p[layer][0].squeeze(0)
        else:
            p = None
        self.p = p
        self.indices = indices

    def compute_cluster(self, in_channel: int):
        self.target_in_channel = in_channel
        if self.p is not None:
            p = self.p[in_channel]
        else:
            p = None
        predicts = predict_cluster(self.som, p)
        self.predicts = predicts

    def calculate_cphap(self):
        in_channel = self.target_in_channel
        layer = self.target_layer
        cphaps = []
        uncertainties = []
        for predict in self.predicts:
            try:
                for_mean_variance = torch.stack(
                    [p_ for p_ in self.hap_lists[layer][in_channel] if (predict == predict_cluster(self.som, p_)).all()]
                )
            except RuntimeError:
                pass
            else:
                cphaps.append(for_mean_variance.mean(0))
                uncertainties.append(for_mean_variance.var(0))

        return cphaps, uncertainties

    def plot(self, cphaps, uncertainties):
        plt.plot(self.x_test[self.data_idx][self.target_in_channel].cpu().numpy())
        if len(cphaps) != 0:
            for i in range(len(cphaps)):
                plt.fill_between(
                    self.indices[i],
                    (cphaps[i] - uncertainties[i]).cpu().numpy(),
                    (cphaps[i] + uncertainties[i]).cpu().numpy(),
                    alpha=0.3,
                    color="g"       # TODO クラスタ毎に色分け
                )

    def reset_p(self):
        self.p = [list() for _ in range(self.model.depth)]


def main(data_idx: int, layer: int, in_channel, target_channel: int, epochs: int):
    cphaps = []
    uncertainties = []
    # hap_listはx_testに対するもの
    hap_lists, (x_train, x_test), model = run_hap()
    som = train_som(hap_lists[layer][0], (8, 8), epochs)
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

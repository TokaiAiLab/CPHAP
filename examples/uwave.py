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

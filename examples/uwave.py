from comet_ml import Experiment
from typing import Tuple, Union
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


def train_nn(
        dataset: str, batch_size: int, depth: int, epochs: int
) -> Tuple[CNN, Tuple[Union[np.ndarray, np.ndarray], Union[np.ndarray, np.ndarray]], Tuple[
    Union[np.ndarray, np.ndarray], Union[np.ndarray, np.ndarray]]]:
    experiment = Experiment(project_name="cphap", auto_output_logging=False)
    experiment.add_tag(dataset)
    experiment.add_tag("NN-depth-{}".format(depth))
    (x_train, y_train), (x_test, y_test) = fetch_dataset(dataset)
    scaler = TimeSeriesScalerMeanVariance()
    x_train: np.ndarray = scaler.fit_transform(x_train)
    x_test: np.ndarray = scaler.transform(x_test)

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

    return runner.model.eval(), (x_train, x_test), (y_train, y_test)


def run_hap(batch_size, depth, dataset_name):
    """
    j == layer
    hap_list[j].shape == [in_channels, sub_seq, rf]

    """
    # dataset_name = "UWaveGestureLibraryAll"
    # dataset_name = "RacketSports"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (x_train, x_test), (y_train, y_test) = train_nn(dataset_name, batch_size, depth, epochs=100)
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

    return hap_lists, (x_train, x_test), model, (y_train, y_test)


class CPHAPFrontend:
    def __init__(self, dataset_name, batch_size, depth, som_map_size: Tuple[int, int] = (8, 8)):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
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
        self.target_nn_channel = None
        colors = []
        for i in range(som_map_size[0]):
            for j in range(som_map_size[1]):
                colors.append([i, j])

        self.colors = np.vstack(colors)
        self.cm = plt.cm.get_cmap("hsv")

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.model_depth = depth

    def train_nn(self):
        self.hap_lists, (self.x_train, self.x_test), self.model, (self.y_train, self.y_test) = run_hap(
            dataset_name=self.dataset_name, batch_size=self.batch_size, depth=self.model_depth
        )
        self.reset_p()

    def train_som(self, layer: int, epoch: int):
        self.target_layer = layer
        self.som = train_som(self.hap_lists[layer][0], self.som_map_size, epoch)

    def compute_p(self, data_idx, target_channel):
        self.data_idx = data_idx
        self.target_nn_channel = target_channel
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
        print("{} activation points were detected in the data.".format(len(self.predicts)))
        for predict in self.predicts:
            start = datetime.datetime.now()
            # for_mean_variance = torch.stack([
            #   p_ for p_ in self.hap_lists[layer][in_channel] if (predict == predict_cluster(self.som, p_)).all()
            # ])
            som_preds = self.som.predict(self.hap_lists[layer][in_channel])[0]
            flags = (predict == som_preds).all(1)
            for_mean_variance = self.hap_lists[layer][in_channel][flags]
            if not flags.any():
                pass
            else:
                cphaps.append(for_mean_variance.mean(0))
                uncertainties.append(for_mean_variance.var(0))
            end = datetime.datetime.now() - start
            print("It took {}s to process each one.".format(end.total_seconds()))

        return cphaps, uncertainties

    def plot(self, cphaps, uncertainties):
        pred, logit = self._nn_predict()
        title = "CNN Layer {} - Channel {} | Input Channel {} | Label: {} | CNN Prediction: {} ({:.4f})".format(
            self.target_layer, self.target_nn_channel, self.target_in_channel, self.y_test[self.data_idx], pred, logit
        )

        plt.figure(figsize=(10, 5))
        plt.plot(self.x_test[self.data_idx][self.target_in_channel].cpu().numpy())
        if len(cphaps) != 0:
            for i in range(len(cphaps)):
                color = self.cm(
                    (self.colors == self.predicts[i]).all(1).argmax() / (self.som_map_size[0] * self.som_map_size[1])
                )
                plt.fill_between(
                    self.indices[i],
                    (cphaps[i] - uncertainties[i]).cpu().numpy(),
                    (cphaps[i] + uncertainties[i]).cpu().numpy(),
                    alpha=0.3,
                    color=color,
                    label=str(self.predicts[i])
                )
            plt.legend()
        plt.title(title)

    def reset_p(self):
        self.p = [list() for _ in range(self.model.depth)]

    def _nn_predict(self):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x: torch.Tensor = self.x_test[self.data_idx].to(device).unsqueeze(0)
            model = self.model.to(device)
            logit = torch.softmax(model(x), 1)
            pred = logit.argmax().cpu().numpy()
            return pred, logit.max().cpu().numpy()

    def save(self, path):
        hap_lists = list(map(lambda x: x.cpu(), self.hap_lists))
        x_train = self.x_train.cpu()
        x_test = self.x_test.cpu()
        y_train = self.y_train
        y_test = self.y_test
        model = self.model.cpu().state_dict()
        model_init = {
            "in_features": self.model.in_features,
            "mid_features": self.model.mid_features,
            "n_class": self.model.n_class,
            "depth": self.model.depth
        }

        torch.save({
            "hap_lists": hap_lists,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "model": model,
            "init": model_init
        }, path)

    def load(self, file):
        checkpoints = torch.load(file)
        init = checkpoints["init"]
        self.model = CNN(init["in_features"], init["mid_features"], init["n_class"], init["depth"])
        self.model.load_state_dict(checkpoints["model"])
        self.hap_lists = checkpoints["hap_lists"]
        self.x_train = checkpoints["x_train"]
        self.x_test = checkpoints["x_test"]
        self.y_train = checkpoints["y_train"]
        self.y_test = checkpoints["y_test"]

        self.reset_p()


def main(data_idx: int, layer: int, in_channel, target_channel: int, epochs: int):
    cphaps = []
    uncertainties = []
    # hap_listはx_testに対するもの
    hap_lists, (x_train, x_test), model, _ = run_hap()
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


def run_frontends():
    for depth in [2, 4, 6, 8, 10]:
        print("Depth: ", depth)
        frontend = CPHAPFrontend("UWaveGestureLibraryAll", batch_size=512, depth=depth)
        frontend.train_nn()
        frontend.save("UWave_depth_{}.pth".format(depth))


if __name__ == '__main__':
    run_frontends()

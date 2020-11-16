from comet_ml import Experiment

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.style
from quicksom.som import SOM
from enchanter.engine.modules import fix_seed
from enchanter.tasks import TimeSeriesUnsupervisedRunner
from sklearn.svm import SVC

from cphap.functions import find_t, han, generate_indices_for_hap
from cphap.receptive_field import receptive_field
from cphap.utils import TSULUEADataset


fix_seed(0)
matplotlib.style.use("seaborn")


def gen_table():
    colors = []
    for i in range(8):
        for j in range(8):
            colors.append([i, j])

    return np.vstack(colors)


class CNN(nn.Module):
    def __init__(self, in_features, mid_features, n_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, mid_features, 3, padding=(3 - 1) * 1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(mid_features, n_class)

    def forward(self, x):
        batch = x.shape[0]
        out = self.conv(x)
        out = self.pool(out)
        out = out.reshape(batch, -1)
        return self.fc(out)


def fetch_loader(batch_size: int, name="RacketSports"):
    train_ds = TSULUEADataset(train=True, name=name)
    test_ds = TSULUEADataset(train=False, name=name)
    n_targets = train_ds.n_targets
    n_features = train_ds.n_features

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("Dataset labels: {}".format(train_ds.projector))

    return train_loader, test_loader, (n_targets, n_features)


def main():
    experiment = Experiment(project_name="cphap", auto_output_logging=False)
    train_loader, test_loader, (n_targets, n_features) = fetch_loader(batch_size=32)
    model = CNN(n_features, 32, n_targets)
    optimizer = optim.Adam(model.parameters())

    runner = TimeSeriesUnsupervisedRunner(
        model, optimizer, experiment, evaluator=SVC(probability=True)
    )
    runner.add_loader("train", train_loader).add_loader("test", test_loader)
    runner.train_config(epochs=1000)
    runner.run()

    receptive_field_dict = receptive_field(model.conv, (n_features, 30))

    data = test_loader.dataset.data
    max_size = data.shape[-1]
    model = runner.model.cpu()
    runner.device = torch.device("cpu")

    def func(c: int, d, padding: int):
        with torch.no_grad():
            all_thresholds = {}
            model.eval()
            hap_list = []
            d = nn.functional.pad(d, [0, padding])
            for idx in range(d.shape[0]):
                # for j in range(n_layers):
                #   hap_list_j = []
                out_j = model.conv(d[idx].unsqueeze(0))
                t_jk = find_t(model.conv(data))  # shape == [channels, L']
                for k in range(n_features):
                    han_jk, _ = han(out_j, k, t_jk)

                    irf_j = int(receptive_field_dict["1"]["r"])
                    tmp = generate_indices_for_hap(han_jk, irf_j, max_size)
                    # hap_list_j.append(sorted(set(tmp)))
                    # hap_list.append(hap_list_j)
                    hap_list.append(d[idx, c, tmp])  # List[List[int]]
                    # data.shape == [batch, in_features, seq_len]
                    # hap_list[0].shape == [in_features, samples, receptive_filed_size_of_layer_j]

        return torch.cat([i for i in hap_list])

    hap_lists = Parallel(n_jobs=-1)([delayed(func)(i, data, 2) for i in range(n_features)])

    def train_som(niter, idx):
        print("IDX: ", idx)
        som = SOM(8, 8, hap_lists[idx].shape[1], niter=niter, device=runner.device)
        learning_error = som.fit(hap_lists[idx].to(runner.device))
        return som

    soms = []
    for i in range(n_features):
        soms.append(train_som(1000, i))

    for i in np.random.choice(list(range(len(test_loader.dataset.data))), size=20):
        x = test_loader.dataset.data[i].unsqueeze(0)
        x = nn.functional.pad(x, [0, 2])
        y = test_loader.dataset.target[i]
        model.eval()
        with torch.no_grad():
            model_logit = runner.encode(x)
            model_pred = runner.evaluator.predict(model_logit)
            model_logit = runner.evaluator.predict_proba(model_logit)[0][model_pred]

            t_jk = find_t(
                model.conv(test_loader.dataset.data)
            )  # shape == [channels, L']
            out = model.conv(x)
            h, m = han(out, 0, t_jk)
            rf = int(receptive_field_dict["1"]["r"])
            hap = generate_indices_for_hap(h, rf, m)
            print("HAP: {} | Idx: {}".format(hap, i))
            ps = []

            cphaps = []
            uncertainties = []
            clusters = []
            for h in hap:
                ps.append(x[0][0][h])
            try:
                ps = torch.stack(ps).to(runner.device)
            except RuntimeError:
                pass

            else:
                predicts, _ = soms[0].predict(ps)

                tmp = torch.cat(hap_lists)

                for predict in predicts:
                    for_mean_variance: torch.Tensor = torch.stack(
                        [p for p in tmp if (soms[0].predict(p.unsqueeze(0).to(runner.device))[0] == predict).all()]
                    )

                    cphaps.append(for_mean_variance.mean(0))
                    uncertainties.append(torch.var(for_mean_variance, dim=0))
                    clusters.append(predict)

        def plot(channel: int):
            cm = plt.cm.get_cmap("hsv")
            table = gen_table()
            plt.figure(figsize=(10, 5))
            plt.plot(x[0][channel])

            title = "Label: {} | Channel : {} | SVM Pred: {} ({:.4f})".format(
                y.cpu().numpy(), channel, model_pred[0], model_logit[0]
            )

            if len(cphaps) != 0:
                for idx in range(len(cphaps)):
                    color = cm((table == clusters[idx]).all(1).argmax() / 64)
                    plt.fill_between(
                        hap[idx], cphaps[idx] - uncertainties[idx], cphaps[idx] + uncertainties[idx],
                        alpha=0.3, color=color, label=str(clusters[idx])
                    )

                plt.legend()
            plt.title(title)

        plot(0)
        plt.show()

    runner.quite()


main()

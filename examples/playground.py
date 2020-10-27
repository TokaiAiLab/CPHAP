import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from quicksom.som import SOM
from torch.utils.data import DataLoader

from cphap.functions import find_t, plot_kde, han, generate_indices_for_hap, calc_half_size
from cphap.receptive_field import receptive_field
from cphap.utils import UEADataset

n_channels = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


class CNN(nn.Module):
    def __init__(self, ins):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(ins, n_channels, 3), nn.ReLU())
        self.fc = nn.Linear(10 * 28, 4)

    def forward(self, x):
        batch = x.shape[0]
        out = self.conv(x)
        out = out.reshape(batch, -1)
        return self.fc(out)


in_features = 6
model = CNN(in_features).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_ds = UEADataset(train=True, name="RacketSports")
test_ds = UEADataset(train=False, name="RacketSports")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


for epoch in range(100):
    for batch in train_loader:
        data, targets = batch
        optimizer.zero_grad()
        logit = model(data)
        loss = criterion(logit, targets)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Loss: ", loss.item())

receptive_field_dict = receptive_field(model.conv, (in_features, 30))


def func(c: int, d):
    d = torch.tensor(d, dtype=torch.float32)
    all_thresholds = {}
    model.eval()
    with torch.no_grad():
        hap_list = []
        for idx in range(d.shape[0]):
            # for j in range(n_layers):
            #   hap_list_j = []
            out_j = model.conv(d[idx].unsqueeze(0))
            t_jk = find_t(model.conv(data))  # shape == [channels, L']
            for k in range(n_channels):
                han_jk, max_size = han(out_j, k, t_jk)

                irf_j = int(receptive_field_dict["1"]["r"])
                tmp = generate_indices_for_hap(han_jk, irf_j, max_size)
                # hap_list_j.append(sorted(set(tmp)))
                # hap_list.append(hap_list_j)
                hap_list.append(d[idx, c, tmp])  # List[List[int]]
                # data.shape == [batch, in_features, seq_len]
                # hap_list[0].shape == [in_features, samples, receptive_filed_size_of_layer_j]

    return torch.cat([i for i in hap_list])


# size of hap_list == data.shape[0] * n_channels
hap_lists = Parallel(n_jobs=-1)([delayed(func)(i, test_ds.data) for i in range(in_features)])
NITER = 1000


def train_som(idx):
    print("IDX: ", idx)
    som = SOM(8, 8, hap_lists[idx].shape[1], niter=NITER, device=device)
    learning_error = som.fit(hap_lists[idx])
    return som


soms = Parallel(n_jobs=-1)([delayed(train_som)(i) for i in range(in_features)])


x = torch.tensor(test_ds.data[0], dtype=torch.float32).unsqueeze(0)
model.eval()
with torch.no_grad():
    t_jk = find_t(model.conv(torch.tensor(test_ds.data, dtype=torch.float32)))  # shape == [channels, L']
    out = model.conv(x)
    h, m = han(out, 0, t_jk)
    rf = int(receptive_field_dict["1"]["r"])
    hap = generate_indices_for_hap(h, rf, m)
    ps = []
    for h in hap:
        ps.append(x[0][0][h])

    ps = torch.stack(ps)
    predicts, _ = soms[0].predict(ps)
    for_mean_variance: torch.Tensor = torch.stack(
        [p for p in hap_lists[0] if (soms[0].predict(p.unsqueeze(0))[0] == predicts[0]).all()]
    )
    # TODO: predicts は複数個含まれることが考えられる。それに対する処理を入れる。
    cphap = for_mean_variance.mean(0)
    uncertainty = torch.var(for_mean_variance, dim=0)


def plot(channel: int):
    plt.figure(figsize=(10, 5))
    plt.plot(x[0][channel])
    plt.fill_between(hap[0], cphap - uncertainty, cphap + uncertainty, alpha=0.3, color="g")

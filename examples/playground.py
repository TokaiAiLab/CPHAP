import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from quicksom.som import SOM

from cphap.functions import find_t, plot_kde, han, generate_indices_for_hap, calc_half_size
from cphap.receptive_field import receptive_field

n_channels = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(5, n_channels, 3), nn.ReLU())
        self.fc = nn.Linear(10 * 98, 5)

    def forward(self, x):
        batch = x.shape[0]
        out = self.conv(x)
        out = out.reshape(batch, 10 * 98)
        return self.fc(out)


model = CNN()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

data = torch.randn(32, 5, 100)
target = torch.randint(0, high=4, size=(32,))

for epoch in range(1000):
    optimizer.zero_grad()
    logit = model(data)
    loss = criterion(logit, target)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("Loss: ", loss.item())

receptive_field_dict = receptive_field(model.conv, (5, 100))


def func(c: int):
    model.eval()
    with torch.no_grad():
        hap_list = []
        for idx in range(data.shape[0]):
            # for j in range(n_layers):
            #   hap_list_j = []
            out_j = model.conv(data[idx].unsqueeze(0))
            t_jk = find_t(model.conv(data))  # shape == [channels, L']
            for k in range(n_channels):
                han_jk, max_size = han(out_j, k, t_jk)

                irf_j = int(receptive_field_dict["1"]["r"])
                tmp = generate_indices_for_hap(han_jk, irf_j, max_size)
                # hap_list_j.append(sorted(set(tmp)))
                # hap_list.append(hap_list_j)
                hap_list.append(data[idx, c, tmp])  # List[List[int]]
                # data.shape == [batch, in_features, seq_len]
                # hap_list[0].shape == [in_features, samples, receptive_filed_size_of_layer_j]

    return torch.cat([i for i in hap_list])


# size of hap_list == data.shape[0] * n_channels
sub = Parallel(n_jobs=-1)([delayed(func)(i) for i in range(5)])
som = SOM(8, 8, sub[0].shape[1], niter=10, device=device)
learning_error = som.fit(sub[0])
pred, error = som.predict(sub[0])

print(pred)
from typing import Tuple, Union

import torch
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


__all__ = ["find_t", "plot_kde", "han"]


def find_t(
    feature_map: torch.Tensor, margin: float = 0.05, return_raw: bool = False
) -> Union[Tuple[np.ndarray, float, gaussian_kde], float]:
    """

    Args:
        feature_map: output of convolution. shape == [1, seq_len]
        margin: default: 0.05
        return_raw: default - True

    Returns:
        - xs ...
        - idx ...
        - kernel ...

    Examples:
         >>> model: torch.nn.Module = ...
         >>> inputs = torch.randn(1, 5, 100)     # [Batch, Channels, Seq Len]
         >>> mid = model.conv(inputs)            # shape == [1, conv_out_channels, Seq Len']
         >>> target_channel: int = 0
         >>> x, i, k = find_t(mid[:, target_channel])
         >>> T_j_k = x[i]

    References:
        https://teratail.com/questions/242195

    """
    with torch.no_grad():
        min_value: int = feature_map.min().item()
        max_value: int = feature_map.max().item()
        seq_len: int = feature_map.shape[1]

        kernel = gaussian_kde(feature_map)  # カーネル密度推定
        xs = np.linspace(min_value, max_value, seq_len ** 2)  # 全区間積分の範囲
        ys = kernel(xs)

        cdf = cumtrapz(ys, xs)  # 積分して累積分布関数を計算
        idx = np.searchsorted(cdf, 1 - margin)
        print(xs[idx])

    if return_raw:
        return xs, idx.item(), kernel
    else:
        return xs[idx].item()  # threshold


def plot_kde(feature_map: torch.Tensor, plot: bool = True) -> None:
    xs, idx, kernel = find_t(feature_map, return_raw=True)
    ys = kernel(xs)
    min_value: int = feature_map.min().item()
    max_value: int = feature_map.max().item()
    plt.plot(xs, kernel(xs))
    plt.fill_between(xs[:idx], 0, ys[:idx])
    plt.xlim(min_value, max_value)
    if plot:
        plt.show()


def han(feature_map: torch.Tensor, margin: float = 0.05):
    threshold = find_t(feature_map, margin)
    _, han_values = np.where(feature_map > threshold)
    return han_values

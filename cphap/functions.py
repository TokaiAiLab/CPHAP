from typing import Tuple, Union, Optional

import torch
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


__all__ = ["find_t", "plot_kde", "han"]


def find_t(
    feature_map: torch.Tensor, margin: float = 0.05, return_raw: bool = False
) -> torch.Tensor:
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
        feature_map = torch.flatten(feature_map)
        values, indices = feature_map.sort()
        threshold_idx: int = int(len(values) * (1.0 - margin))
        threshold = values[threshold_idx]

    return threshold


def plot_kde(feature_map: torch.Tensor, plot: bool = True, margin: float = 0.05) -> None:
    tmp = feature_map
    with torch.no_grad():
        feature_map = torch.flatten(feature_map)
        min_value: int = feature_map.min().item()
        max_value: int = feature_map.max().item()

        kernel = gaussian_kde(feature_map)  # カーネル密度推定
        xs = np.linspace(min_value, max_value, 5000)  # 全区間積分の範囲
        ys = kernel(xs)

    threshold = find_t(tmp).numpy()
    plt.plot(xs, kernel(xs))

    plt.fill_between(xs[xs > threshold], 0, ys[xs > threshold])
    plt.xlim(min_value, max_value)
    if plot:
        plt.show()


def han(feature_map: torch.Tensor, margin: float = 0.05):
    # feature_map.shape == [N, L]
    threshold = find_t(feature_map, margin)
    if threshold:
        all_values = []
        for sample in range(feature_map.shape[0]):
            values = {i for i in range(feature_map.shape[-1]) if feature_map[sample, i] > threshold}
            all_values.append(values)
    # _, han_values = np.where(feature_map > threshold)
        return all_values
    else:
        raise Exception


def hap(han_result, conv_in):
    return conv_in[han_result]

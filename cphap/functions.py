from typing import Tuple, Union, Optional, List, Set

import torch
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


__all__ = ["find_t", "plot_kde", "han"]


def find_t(feature_map: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """

    Args:
        feature_map: output of convolution. shape == [1, seq_len]
        margin: default: 0.05

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
        feature_map = feature_map.permute(1, 0, 2).flatten(1)
        values, indices = feature_map.sort()
        threshold_idx: int = int(len(values[0]) * (1.0 - margin))
        threshold = values[:, threshold_idx]

    return threshold


def plot_kde(
    feature_map: torch.Tensor, plot: bool = True, margin: float = 0.05
) -> None:
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


def han(conv_out: torch.Tensor, channel: int, thresholds: List[int]) -> Set[int]:
    a_jk: torch.Tensor = conv_out[0][channel]
    han_jk = {i for i in range(a_jk.shape[0]) if a_jk[i] > thresholds[channel]}
    return han_jk


def calc_half_size(file_size: int) -> int:
    if file_size == 1:
        return 0
    else:
        return file_size // 2


def hap(han_jk: Set[int], in_receptive_filed: int, max_size: int) -> List[int]:
    tmp = []
    for i in han_jk:
        half_size = calc_half_size(in_receptive_filed)
        start: int = i - half_size
        if start < 0:
            start = 0

        end: int = i + half_size
        if end > max_size:
            end = max_size

        temporal_indices = set(range(start, end + 1))
        tmp += temporal_indices

    return sorted(set(tmp))

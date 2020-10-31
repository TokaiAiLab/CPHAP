from typing import Tuple, Union, Optional, List, Set

import torch
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


__all__ = ["find_t", "plot_kde", "han", "generate_indices_for_hap"]


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


@torch.jit.script
def han(
    conv_out: torch.Tensor, channel: int, thresholds: torch.Tensor
) -> Tuple[List[int], int]:
    """

    Args:
        conv_out: output of Conv
        channel:
        thresholds:

    Returns:

    """
    a_jk: torch.Tensor = conv_out[0][channel]   # shape == seq_len
    max_size = a_jk.shape[0]    # max seq len
    han_jk: List[int] = []
    for i in range(max_size):
        if a_jk[i] > thresholds[channel]:
            han_jk.append(i)
    return han_jk, max_size


@torch.jit.script
def calc_half_size(filter_size: int) -> int:
    if filter_size == 1:
        return 0
    elif filter_size % 2 == 0:
        raise ValueError
    else:
        return filter_size // 2


@torch.jit.script
def generate_indices_for_hap(
    han_jk: List[int], in_receptive_filed: int, max_size: int
) -> List[torch.Tensor]:
    tmp = []
    for i in han_jk:
        half_size = calc_half_size(in_receptive_filed)
        start: int = i - half_size
        if start < 0:
            start = i
        end = start + in_receptive_filed

        if end > max_size:
            start -= 1
            end -= 1

        temporal_indices = torch.arange(start, end)
        assert (
            len(temporal_indices) == in_receptive_filed
        ), "Size Error: start - {} end - {}".format(start, end)
        tmp.append(temporal_indices)

    return tmp

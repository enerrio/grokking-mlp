from typing import Dict, Any, Tuple
import yaml  # type: ignore
import torch
import torch.nn as nn


def load_config(file_path: str) -> Dict[str, Any]:
    """Load YAML config file"""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def calculate_sparsity(net: nn.Module, threshold: float) -> Tuple[float, int]:
    """Calculate how sparse a given model is (0 is fully dense, 1 is fully sparse)

    Args:
        net (nn.Module): PyTorch model
        threshold (float): Threshold value to define if a weight is sparse or not

    Returns:
        tuple[float, int]: Sparsity ratio i.e. how many weights in net are below threshold and total number of non-sparse weights
    """
    sparse_count = 0
    total_count = 0
    for param in net.parameters():
        sparse_count += int(torch.sum(torch.abs(param) < threshold).item())
        total_count += param.numel()

    sparsity_ratio = sparse_count / total_count
    active_weights = total_count - sparse_count
    return sparsity_ratio, active_weights


def prune_network(net: nn.Module, threshold: float) -> nn.Module:
    """Prune network by setting weights smaller than threshold to 0

    Args:
        net (nn.Module): Fully trained PyTorch model
        threshold (float): Threshold value to define if a weight is sparse or not

    Returns:
        nn.Module: PyTorch model with weights pruned
    """
    # Prune each layer by calculating a binary mask and applying
    for _, param in net.named_parameters():
        prune_mask = (torch.abs(param) < threshold).type(torch.bool)
        param.data.mul_(~prune_mask)
    # sanity check: calculate sparsity and print
    # sparsity = calculate_sparsity(net, threshold)
    # print(f"New sparsity: {sparsity:.2%}")
    return net

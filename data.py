from typing import Dict, Tuple, Optional
import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset


def create_data(
    p: int, operation: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Dict[int, str]]:
    """Create synthetic modular addition dataset"""
    char2idx = {str(i): i for i in range(p)}
    char2idx[operation] = len(char2idx)
    char2idx["="] = len(char2idx)
    idx2char = {v: k for k, v in char2idx.items()}
    # Create dataset for modular addition
    a = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b = einops.repeat(torch.arange(p), "i -> (j i)", j=p)
    y = (a + b) % p
    op = torch.full((p * p,), char2idx[operation])
    eq = torch.full((p * p,), char2idx["="])

    # Combine into a single tensor for features and labels
    features = torch.stack([a, op, b, eq], dim=-1)
    y_ohe = F.one_hot(y, len(idx2char)).to(torch.float32)
    return features, y_ohe, char2idx, idx2char


def create_dataloader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    seed: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[DataLoader, ...]:
    """Split into train/val sets and create PyTorch DataLoader

    Args:
        features (torch.Tensor): Data features used for training
        labels (torch.Tensor): Corresponding labels for the features
        batch_size (int): Number of samples in each batch before weight update step
        seed (int): Integer for setting PyTorch random seed
        mask (torch.Tensor, optional): Boolean mask to apply to features/labels. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: Dataloaders for training and val sets
    """
    # Split into train/val sets
    rng = torch.Generator().manual_seed(seed)

    if isinstance(mask, torch.Tensor):
        # Split data based on provided mask
        train_data = TensorDataset(features[~mask], labels[~mask])
        val_data = TensorDataset(features[mask], labels[mask])

    else:
        # Create train and val set randomly
        data = TensorDataset(features, labels)
        train_data, val_data = random_split(data, [0.8, 0.2], generator=rng)  # type: ignore

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=rng
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, generator=rng
    )
    return train_dataloader, val_dataloader


def convert_to_str(tokens: torch.Tensor, idx2char: dict) -> str:
    """Convert a tensor vector of token ids to string"""
    return " ".join([idx2char[x] for x in tokens.tolist()])

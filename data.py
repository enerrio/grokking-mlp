import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset


def create_data(p, operation):
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


def create_dataloader(features, labels, batch_size, mask=None):
    """Split into train/val sets and create PyTorch DataLoader"""
    # Split into train/val sets
    rng = torch.Generator().manual_seed(21)

    if mask:
        # Split data based on provided mask
        train_data = TensorDataset(features[mask], labels[~mask])
        val_data = TensorDataset(features[~mask], labels[mask])

    else:
        # Create train and val set randomly
        data = TensorDataset(features, labels)
        train_data, val_data = random_split(data, [0.8, 0.2], generator=rng)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=rng
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, generator=rng
    )
    return train_dataloader, val_dataloader


def convert_to_str(tokens, idx2char):
    """Convert a tensor vector of token ids to string"""
    return " ".join([idx2char[x] for x in tokens.tolist()])

import torch
import torch.nn as nn
from net import MLP
from data import create_data, create_dataloader
from utils import load_config, calculate_sparsity, prune_network


def main() -> None:
    config = load_config("config.yaml")
    device = torch.device(config["device"])
    # Load model
    print(f"Loading model from experiment: {config['experiment_name']}")
    vocab_size = config["data"]["p"] + 2
    net = MLP(vocab_size)
    net.load_state_dict(torch.load(f"{config['experiment_name']}/model_wts.pth"))
    net = net.to(device)
    num_params = sum([x.numel() for x in net.parameters()])
    # calculate sparsity
    sparsity_ratio, active_weights = calculate_sparsity(
        net, config["sparsity"]["threshold"]
    )
    print(f"Model sparsity: {sparsity_ratio:.2%}")
    print(f"Total number of weights: {num_params:,}")
    print(f"Total number of active weights: {active_weights:,}")
    # prune model
    pruned_net = prune_network(net, config["sparsity"]["threshold"])
    # evaluate on train/val datasets
    features, y_ohe, _, _ = create_data(
        config["data"]["p"], config["data"]["operation"]
    )
    train_dataloader, val_dataloader = create_dataloader(
        features,
        y_ohe,
        config["model"]["batch_size"],
        config["data"]["random_seeds"][0],
    )
    criterion = nn.CrossEntropyLoss()
    total_correct = total_loss = 0.0
    total_val_correct = total_val_loss = 0.0
    for inputs, labels in train_dataloader:
        # forward pass
        outputs = pruned_net(inputs)
        loss = criterion(outputs, labels)
        # calculate metrics
        total_correct += (
            (outputs.detach().softmax(-1).argmax(-1) == labels.detach().argmax(-1))
            .numpy()
            .sum()
        )
        total_loss += loss.item()
    for inputs, labels in val_dataloader:
        # forward pass
        outputs = pruned_net(inputs)
        loss = criterion(outputs, labels)
        # calculate metrics
        total_val_correct += (
            (outputs.detach().softmax(-1).argmax(-1) == labels.detach().argmax(-1))
            .numpy()
            .sum()
        )
        total_val_loss += loss.item()
    train_acc = total_correct / len(train_dataloader.dataset)  # type: ignore
    train_loss = total_loss / len(train_dataloader)
    val_acc = total_val_correct / len(val_dataloader.dataset)  # type: ignore
    val_loss = total_val_loss / len(val_dataloader)
    print(f"Training loss: {train_loss:.2f}\tTraining accuracy: {train_acc:.2%}")
    print(f"Validation loss: {val_loss:.2f}\tValidation accuracy: {val_acc:.2%}")


if __name__ == "__main__":
    main()

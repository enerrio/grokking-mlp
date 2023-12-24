from tqdm.auto import tqdm
import time
import yaml
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from data import create_data, create_dataloader, convert_to_str
from net import MLP, layer_norm
from plot import plot_train_results, plot_weights


def load_config(file_path):
    """Load YAML config file"""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_loop(
    net,
    optimizer,
    criterion,
    num_epochs,
    train_dataloader,
    val_dataloader,
    log_epoch,
    device,
):
    """Train a provided model for a given number of epochs

    Args:
        net (nn.Module): PyTorch model to train
        optimizer (optim.Optimizer): Pytorch optimizer to use for weight updates
        criterion (nn.Module): Loss function
        num_epochs (int): Total number of epochs to train for
        train_dataloader (torch.utils.data.DataLoader): Dataloader for supplying training data to model
        val_dataloader (torch.utils.data.DataLoader): Dataloader for supplying validation data to model
        log_epoch (int): Every X epochs log training statistics
        device (torch.device): Device that data and model should move to while training

    Returns:
        dict: Statistics for the training run
    """
    train_stats = {
        "epoch_train_losses": [],
        "epoch_train_accs": [],
        "epoch_val_losses": [],
        "epoch_val_accs": [],
        "epoch_layer_norm": [],
    }
    step = 0
    for epoch in tqdm(range(num_epochs)):
        total_correct = total_loss = 0.0
        total_val_correct = total_val_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            step += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.type(torch.float64)  # avoid loss spikes
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Keep track of stats
            with torch.no_grad():
                total_correct += (
                    (
                        outputs.detach().softmax(-1).argmax(-1)
                        == labels.detach().argmax(-1)
                    )
                    .numpy()
                    .sum()
                )
                total_loss += loss.item()

        # Evaluate on validation set
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                # forward pass
                outputs = net(inputs)
                val_loss = criterion(outputs, labels)
                # calculate metrics
                total_val_correct += (
                    (
                        outputs.detach().softmax(-1).argmax(-1)
                        == labels.detach().argmax(-1)
                    )
                    .numpy()
                    .sum()
                )
                total_val_loss += val_loss.item()

        # Store epoch-level loss + accuracy
        epoch_acc = (total_correct / len(train_dataloader.dataset)) * 100.0
        epoch_loss = total_loss / len(train_dataloader)
        val_acc = (total_val_correct / len(val_dataloader.dataset)) * 100.0
        val_epoch_loss = total_val_loss / len(val_dataloader)
        train_stats["epoch_train_accs"].append(epoch_acc)
        train_stats["epoch_train_losses"].append(epoch_loss)
        train_stats["epoch_val_accs"].append(val_acc)
        train_stats["epoch_val_losses"].append(val_epoch_loss)
        train_stats["epoch_layer_norm"].append(layer_norm(net))

        # Log losses every X epochs
        if (epoch % log_epoch) == 0:
            tqdm.write(
                f"[{epoch+1}/{num_epochs}][{step:,}]"
                f"\tTrain Loss: {epoch_loss:.3f}"
                f"\tVal Loss: {val_epoch_loss:.3f}"
            )
        # stop training after about X optimization steps
        # if step > num_optimization_steps:
        #     break
    return train_stats


def main():
    # Load config
    config = load_config("config.yaml")
    device = torch.device(config["device"])

    # Create data
    features, y_ohe, char2idx, idx2char = create_data(
        config["data"]["p"], config["data"]["operation"]
    )

    for i in np.random.randint(0, features.shape[0], size=10):
        print(f"Input tensor & label: {features[i]}, {y_ohe[i].argmax(-1)}")
        print(
            f"Original equation: {convert_to_str(torch.cat([features[i], y_ohe[i].argmax(-1).unsqueeze(0)]), idx2char)}"
        )

    train_stats = {}

    # Run experiments
    start_exps = time.time()
    for seed in config["data"]["random_seeds"]:
        # Get dataloader
        mask = None
        if config["data"]["use_mask"]:
            y = y_ohe.argmax(-1)
            mask = y < 20
        train_dataloader, val_dataloader = create_dataloader(
            features, y_ohe, config["model"]["batch_size"], seed, mask
        )
        print(f"Running experiment for random seed: {seed}")
        print(f"Total samples in dataset: {features.shape[0]:,}")
        print(f"Total samples in train set: {len(train_dataloader.dataset):,}")
        print(f"Total samples in val set: {len(val_dataloader.dataset):,}")

        print(f"Total number of batches in train dataloader: {len(train_dataloader):,}")
        print(f"Total number of batches in val dataloader: {len(val_dataloader):,}")

        # Create model
        vocab_size = config["data"]["p"] + 2
        net = MLP(vocab_size)
        net = net.to(device)

        # Create optimizer + loss func
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            net.parameters(), lr=config["optim"]["lr"], betas=config["optim"]["betas"]
        )
        num_params = sum([x.numel() for x in net.parameters()])
        print(f"Number of parameters in model: {num_params:,}")

        # Train
        start = time.time()
        train_stats[f"seed_{seed}"] = train_loop(
            net,
            optimizer,
            criterion,
            config["model"]["num_epochs"],
            train_dataloader,
            val_dataloader,
            config["model"]["log_epoch"],
            device,
        )
        print(f"Training time: {(time.time()-start)/60.:.2f} minutes")
    print(
        f"Total training time (all seeds): {(time.time()-start_exps)/60.:.2f} minutes"
    )

    # Plot train run
    plot_train_results(
        train_stats,
        config["plot"]["save_plots"],
        config["experiment_name"],
    )

    print(
        f"Performing dimensionality reduction using final model in the experiment. Seed: {seed}"
    )
    # Perform dimensionality reduction on layers and plot
    embed_wt_reduced = PCA(2).fit_transform(net.net[0].weight.data)
    print(f"Shape of embedding weights after PCA: {embed_wt_reduced.shape}")

    output_wt_reduced = PCA(2).fit_transform(
        net(
            features[features[:, 2] == 8]
        ).detach()  # select equations of the form X + 8
    )
    print(f"Shape of output layer after PCA: {output_wt_reduced.shape}")
    plot_weights(
        embed_wt_reduced,
        output_wt_reduced,
        features[features[:, 2] == 8][:, 0],  # Overlay X from equation X + 8
        char2idx,
        config["plot"]["save_plots"],
        config["experiment_name"],
    )
    print("Saving final model in the experiment...")
    torch.save(net.state_dict(), f"{config['experiment_name']}/model_wts.pth")
    print("Complete!")


if __name__ == "__main__":
    main()

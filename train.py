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


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create data
    features, y_ohe, char2idx, idx2char = create_data(
        config["data"]["p"], config["data"]["operation"]
    )

    for i in np.random.randint(0, features.shape[0], size=10):
        print(f"Input tensor & label: {features[i]}, {y_ohe[i].argmax(-1)}")
        print(
            f"Original equation: {convert_to_str(torch.cat([features[i], y_ohe[i].argmax(-1).unsqueeze(0)]), idx2char)}"
        )

    # Get dataloader
    mask = None
    if config["data"]["use_mask"]:
        y = y_ohe.argmax(-1)
        mask = (y == 1) | (y == 2)
    train_dataloader, val_dataloader = create_dataloader(
        features, y_ohe, config["model"]["batch_size"], mask
    )
    print(f"Total samples in dataset: {features.shape[0]:,}")
    print(f"Total samples in train set: {len(train_dataloader.dataset):,}")
    print(f"Total samples in val set: {len(val_dataloader.dataset):,}")

    print(f"Total number of batches in train dataloader: {len(train_dataloader):,}")
    print(f"Total number of batches in val dataloader: {len(val_dataloader):,}")

    # Create model
    vocab_size = config["data"]["p"] + 2
    net = MLP(vocab_size)

    # Create optimizer + loss func
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config["optim"]["lr"], betas=config["optim"]["betas"]
    )
    num_params = sum([x.numel() for x in net.parameters()])
    print(f"Number of parameters in model: {num_params:,}")

    # Train
    train_stats = {
        "epoch_train_losses": [],
        "epoch_train_accs": [],
        "epoch_val_losses": [],
        "epoch_val_accs": [],
        "epoch_layer_norm": [],
    }
    start = time.time()

    step = 0
    for epoch in tqdm(range(config["model"]["num_epochs"])):
        total_correct = total_loss = 0.0
        total_val_correct = total_val_loss = 0.0
        for inputs, labels in train_dataloader:
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
        if (epoch % 100) == 0:
            tqdm.write(
                f"[{epoch+1}/{config['model']['num_epochs']}][{step:,}]"
                f"\tTrain Loss: {epoch_loss:.3f}"
                f"\tVal Loss: {val_epoch_loss:.3f}"
            )

        # stop training after about X optimization steps
        # if step > num_optimization_steps:
        #     break
    print(f"Total training time: {(time.time()-start)/60.:.2f} minutes")

    # Plot train run
    plot_train_results(
        train_stats,
        config["plot"]["save_plots"],
        config["experiment_name"],
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
    torch.save(net.state_dict(), f"{config['experiment_name']}/model_wts.pth")


if __name__ == "__main__":
    main()

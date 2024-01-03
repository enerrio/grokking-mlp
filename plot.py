import os
import numpy as np
import plotly.graph_objects as go  # type: ignore

TRAIN_LINE_COLOR = "rgb(101,110,242)"
TRAIN_FILL_COLOR = "rgba(101,110,242,0.2)"
VAL_LINE_COLOR = "rgb(221,96,70)"
VAL_FILL_COLOR = "rgba(221,96,70,0.2)"
CLEAR_LINE_COLOR = "rgba(255,255,255,0)"
LAYER_NORM_LINE_COLOR = "rgb(128,213,186)"
LAYER_NORM_FILL_COLOR = "rgba(128,213,186,0.2)"


def plot_train_results(train_stats: dict, save: bool, save_path: str) -> None:
    """Plot the loss curve and accuracy for a training run"""
    # Calculate loss mean and standard deviation
    stacked_train_losses = np.stack(
        [train_stats[seed]["epoch_train_losses"] for seed in train_stats]
    )
    stacked_val_losses = np.stack(
        [train_stats[seed]["epoch_val_losses"] for seed in train_stats]
    )
    stacked_layer_norms = np.stack(
        [train_stats[seed]["epoch_layer_norm"] for seed in train_stats]
    )
    mean_train_loss = np.mean(stacked_train_losses, axis=0)
    std_train_loss = np.std(stacked_train_losses, axis=0)
    mean_val_loss = np.mean(stacked_val_losses, axis=0)
    std_val_loss = np.std(stacked_val_losses, axis=0)
    mean_layer_norm = np.mean(stacked_layer_norms, axis=0)
    std_layer_norm = np.std(stacked_layer_norms, axis=0)
    epochs = np.arange(stacked_train_losses.shape[-1])

    fig_loss = go.Figure()

    # Plot loss confidence interval areas
    fig_loss.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_train_loss - std_train_loss, (mean_train_loss + std_train_loss)[::-1]]
        ).clip(
            0
        ),  # clip loss to 0
        fill="toself",
        fillcolor=TRAIN_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Train Confidence Interval",
    )
    fig_loss.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_val_loss - std_val_loss, (mean_val_loss + std_val_loss)[::-1]]
        ).clip(
            0
        ),  # clip loss to 0
        fill="toself",
        fillcolor=VAL_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Val Confidence Interval",
    )
    # Plot mean loss curve + loss curves for all random seeds
    fig_loss.add_scatter(
        x=epochs,
        y=mean_train_loss,
        marker=dict(color=TRAIN_LINE_COLOR),
        name="Train Loss",
    )
    fig_loss.add_scatter(
        x=epochs,
        y=mean_val_loss,
        marker=dict(color=VAL_LINE_COLOR),
        name="Val Loss",
    )
    for seed in train_stats:
        fig_loss.add_scatter(
            x=epochs,
            y=train_stats[seed]["epoch_train_losses"],
            marker=dict(color=TRAIN_LINE_COLOR),
            name=f"Train {seed}",
            opacity=0.3,
            showlegend=False,
        )
        fig_loss.add_scatter(
            x=epochs,
            y=train_stats[seed]["epoch_val_losses"],
            marker=dict(color=VAL_LINE_COLOR),
            name=f"Val {seed}",
            opacity=0.3,
            showlegend=False,
        )
    # Plot mean layer norms
    fig_loss.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_layer_norm - std_layer_norm, (mean_layer_norm + std_layer_norm)[::-1]]
        ),
        fill="toself",
        fillcolor=LAYER_NORM_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Output Layer Norm Confidence Interval",
    )
    fig_loss.add_scatter(
        x=epochs,
        y=mean_layer_norm,
        marker=dict(color=LAYER_NORM_LINE_COLOR),
        name="Output Layer Norm",
    )
    # Add buttons for scaling axis
    updatemenus_loss = [
        dict(
            type="buttons",
            direction="left",
            buttons=list(
                [
                    dict(
                        args=[{"yaxis": dict(type="linear")}],
                        label="Y Axis Linear Scale",
                        method="relayout",
                    ),
                    dict(
                        args=[{"yaxis": dict(type="log")}],
                        label="Y Axis Log Scale",
                        method="relayout",
                    ),
                ]
            ),
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.08,
            yanchor="top",
        )
    ]
    fig_loss.update_layout(
        height=700,
        width=1000,
        title_text="Mean Loss Curve with Confidence Interval",
        title_x=0.5,
        xaxis_title_text="epoch",
        yaxis_title_text="loss",
    )

    # Calculate accuracy mean and standard deviation
    stacked_train_accs = np.stack(
        [train_stats[seed]["epoch_train_accs"] for seed in train_stats]
    )
    stacked_val_accs = np.stack(
        [train_stats[seed]["epoch_val_accs"] for seed in train_stats]
    )
    mean_train_acc = np.mean(stacked_train_accs, axis=0)
    std_train_acc = np.std(stacked_train_accs, axis=0)
    mean_val_acc = np.mean(stacked_val_accs, axis=0)
    std_val_acc = np.std(stacked_val_accs, axis=0)

    fig_acc = go.Figure()

    # Plot accuracy confidence interval areas
    fig_acc.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_train_acc - std_train_acc, (mean_train_acc + std_train_acc)[::-1]]
        ).clip(0, 100),
        fill="toself",
        fillcolor=TRAIN_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Train Confidence Interval",
    )
    fig_acc.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_val_acc - std_val_acc, (mean_val_acc + std_val_acc)[::-1]]
        ).clip(0, 100),
        fill="toself",
        fillcolor=VAL_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Val Confidence Interval",
    )
    # Plot train + val accuracy
    fig_acc.add_scatter(
        x=epochs,
        y=mean_train_acc,
        marker=dict(color=TRAIN_LINE_COLOR),
        name="Train Accuracy",
    )
    fig_acc.add_scatter(
        x=epochs,
        y=mean_val_acc,
        marker=dict(color=VAL_LINE_COLOR),
        name="Val Accuracy",
    )
    for seed in train_stats:
        fig_acc.add_scatter(
            x=epochs,
            y=train_stats[seed]["epoch_train_accs"],
            marker=dict(color=TRAIN_LINE_COLOR),
            name=f"Train {seed}",
            opacity=0.3,
            showlegend=False,
        )
        fig_acc.add_scatter(
            x=epochs,
            y=train_stats[seed]["epoch_val_accs"],
            marker=dict(color=VAL_LINE_COLOR),
            name=f"Val {seed}",
            opacity=0.3,
            showlegend=False,
        )
    # Plot mean layer norms
    fig_acc.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_layer_norm - std_layer_norm, (mean_layer_norm + std_layer_norm)[::-1]]
        ),
        fill="toself",
        fillcolor=LAYER_NORM_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Output Layer Norm Confidence Interval",
    )
    fig_acc.add_scatter(
        x=epochs,
        y=mean_layer_norm,
        marker=dict(color=LAYER_NORM_LINE_COLOR),
        name="Output Layer Norm",
    )
    # Add buttons for scaling axis
    updatemenus_acc = [
        dict(
            type="buttons",
            direction="left",
            buttons=list(
                [
                    dict(
                        args=[{"xaxis": dict(type="linear")}],
                        label="X Axis Linear Scale",
                        method="relayout",
                    ),
                    dict(
                        args=[{"xaxis": dict(type="log")}],
                        label="X Axis Log Scale",
                        method="relayout",
                    ),
                ]
            ),
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.08,
            yanchor="top",
        )
    ]
    # Update layout
    fig_acc.update_layout(
        height=700,
        width=1000,
        title_text="Mean Accuracy with Confidence Interval",
        title_x=0.5,
        xaxis_title_text="epoch",
        yaxis_title_text="accuracy",
    )

    if save:
        os.makedirs(save_path, exist_ok=True)
        # save loss plots
        fig_loss.update_layout(yaxis_type="log")
        fig_loss.write_image(f"{save_path}/losslog.png")
        fig_loss.update_layout(yaxis_type="linear")
        fig_loss.write_image(f"{save_path}/losslinear.png")
        # save accuracy plots
        fig_acc.update_layout(xaxis_type="log")
        fig_acc.write_image(f"{save_path}/acclog.png")
        fig_acc.update_layout(xaxis_type="linear")
        fig_acc.write_image(f"{save_path}/acclinear.png")

    # Update layouts and show in browser
    fig_loss.update_layout(updatemenus=updatemenus_loss)
    fig_acc.update_layout(updatemenus=updatemenus_acc)
    fig_loss.show()
    fig_acc.show()


def plot_weights(
    embed_wt_reduced: np.ndarray,
    output_wt_reduced: np.ndarray,
    output_wt_text: np.ndarray,
    char2idx: dict,
    save: bool,
    save_path: str,
) -> None:
    """Plot layer weights after reducing dimensions"""
    fig_weights = go.Figure()
    fig_weights.add_scatter(
        x=embed_wt_reduced[:, 0],
        y=embed_wt_reduced[:, 1],
        mode="markers+text",
        text=list(char2idx.keys()),
        textposition="top center",
        visible=True,
        name="embed",
    )
    fig_weights.add_scatter(
        x=output_wt_reduced[:, 0],
        y=output_wt_reduced[:, 1],
        mode="markers+text",
        text=output_wt_text,
        textposition="top center",
        visible=False,
        name="output",
    )

    # Add buttons for switching layers
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            buttons=list(
                [
                    dict(
                        args=["visible", [True, False]],
                        label="Embedding layer",
                        method="restyle",
                    ),
                    dict(
                        args=["visible", [False, True]],
                        label="Output layer",
                        method="restyle",
                    ),
                ]
            ),
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.08,
            yanchor="top",
        )
    ]

    # Update layout
    fig_weights.update_layout(
        height=700,
        width=1000,
        title_text="Layer weights",
        title_x=0.5,
    )

    if save:
        os.makedirs(save_path, exist_ok=True)
        fig_weights.update_traces(visible=False, selector=dict(name="embed"))
        fig_weights.update_traces(visible=True, selector=dict(name="output"))
        fig_weights.write_image(f"{save_path}/wts_dimred_output.png")
        fig_weights.update_traces(visible=True, selector=dict(name="embed"))
        fig_weights.update_traces(visible=False, selector=dict(name="output"))
        fig_weights.write_image(f"{save_path}/wts_dimred_embed.png")

    fig_weights.update_layout(updatemenus=updatemenus)
    fig_weights.show()

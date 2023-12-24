import os
import numpy as np
import plotly.graph_objects as go

TRAIN_LINE_COLOR = "rgb(101,110,242)"
TRAIN_FILL_COLOR = "rgba(101,110,242,0.2)"
VAL_LINE_COLOR = "rgb(221,96,70)"
VAL_FILL_COLOR = "rgba(221,96,70,0.2)"
CLEAR_LINE_COLOR = "rgba(255,255,255,0)"


def plot_train_results(train_stats, save, save_path):
    """Plot the loss curve and accuracy for a training run"""
    # Calculate loss mean and standard deviation
    stacked_train_losses = np.stack(
        [train_stats[seed]["epoch_train_losses"] for seed in train_stats]
    )
    stacked_val_losses = np.stack(
        [train_stats[seed]["epoch_val_losses"] for seed in train_stats]
    )
    mean_train_loss = np.mean(stacked_train_losses, axis=0)
    std_train_loss = np.std(stacked_train_losses, axis=0)
    mean_val_loss = np.mean(stacked_val_losses, axis=0)
    std_val_loss = np.std(stacked_val_losses, axis=0)
    epochs = np.arange(stacked_train_losses.shape[-1])

    fig_loss = go.Figure()

    # Add loss confidence interval areas
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
    # Plot mean loss curve
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

    # TODO: TEST plotting layer norm
    # fig_loss.add_scatter(
    #     x=np.arange(len(train_stats["epoch_layer_norm"])),
    #     y=train_stats["epoch_layer_norm"],
    #     name="Output layer norm",
    # )
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

    # Calculate loss mean and standard deviation
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

    # Add accuracy confidence interval areas
    fig_acc.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_train_acc - std_train_acc, (mean_train_acc + std_train_acc)[::-1]]
        ),
        # ).clip(
        #     0, 100
        # ),
        fill="toself",
        fillcolor=TRAIN_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Train Confidence Interval",
    )
    fig_acc.add_scatter(
        x=np.concatenate([epochs, epochs[::-1]]),
        y=np.concatenate(
            [mean_val_acc - std_val_acc, (mean_val_acc + std_val_acc)[::-1]]
        ),
        # ).clip(
        #     0, 100
        # ),
        fill="toself",
        fillcolor=VAL_FILL_COLOR,
        line=dict(color=CLEAR_LINE_COLOR),
        name="Val Confidence Interval",
    )

    # Plot accuracy
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

    # TODO: test plotting layer norm
    # fig_acc.add_scatter(
    #     x=np.arange(len(train_stats["epoch_layer_norm"])),
    #     y=train_stats["epoch_layer_norm"],
    #     name="Output layer norm",
    # )
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
    embed_wt_reduced, output_wt_reduced, output_wt_text, char2idx, save, save_path
):
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

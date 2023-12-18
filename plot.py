import os
import numpy as np
import plotly.graph_objects as go


def plot_train_results(train_stats, save, save_path):
    """Plot the loss curve and accuracy for a training run"""
    # Plot loss curve
    fig_loss = go.Figure()
    fig_loss.add_scatter(
        x=np.arange(len(train_stats["epoch_train_losses"])),
        y=train_stats["epoch_train_losses"],
        name="Train Loss",
    )
    fig_loss.add_scatter(
        x=np.arange(len(train_stats["epoch_val_losses"])),
        y=train_stats["epoch_val_losses"],
        name="Val Loss",
    )
    # TODO: TEST
    fig_loss.add_scatter(
        x=np.arange(len(train_stats["epoch_layer_norm"])),
        y=train_stats["epoch_layer_norm"],
        name="Output layer norm",
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
            y=1.1,
            yanchor="top",
        )
    ]
    fig_loss.update_layout(
        height=700,
        width=1000,
        title_text="Loss Curve",
        title_x=0.5,
        xaxis_title_text="epoch",
        yaxis_title_text="loss",
    )

    # Plot accuracy
    fig_acc = go.Figure()
    fig_acc.add_scatter(
        x=np.arange(len(train_stats["epoch_train_accs"])),
        y=train_stats["epoch_train_accs"],
        name="Train Accuracy",
    )
    fig_acc.add_scatter(
        x=np.arange(len(train_stats["epoch_val_accs"])),
        y=train_stats["epoch_val_accs"],
        name="Val Accuracy",
    )
    fig_acc.add_scatter(
        x=np.arange(len(train_stats["epoch_layer_norm"])),
        y=train_stats["epoch_layer_norm"],
        name="Output layer norm",
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
            y=1.1,
            yanchor="top",
        )
    ]
    # Update layout
    fig_acc.update_layout(
        height=700,
        width=1000,
        title_text="Training Accuracy",
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
            y=1.1,
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

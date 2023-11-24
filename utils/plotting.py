from .diff import DiffusionModel

import plotly.graph_objects as go

def plot_loss(metadata: DiffusionModel.TrainingMetadata):
    """Plots the loss over time.

    Args:
        metadata (DiffusionModel.TrainingMetadata): Training metadata.
    """    
    epochs = sorted(list(metadata.epoch_to_metadata.keys()))

    fig = go.Figure()
    trace = go.Scatter(
        x=epochs, 
        y=[ metadata.epoch_to_metadata[epoch].val_loss for epoch in epochs ], 
        mode="lines",
        name="Validation loss"
        )
    fig.add_trace(trace)

    trace = go.Scatter(
        x=epochs, 
        y=[ metadata.epoch_to_metadata[epoch].train_loss for epoch in epochs ], 
        mode="lines",
        name="Training loss"
        )
    fig.add_trace(trace)

    fig.update_layout(
        title="Loss over time", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        width=800,
        height=600,
        font=dict(size=18)
        )

    return fig
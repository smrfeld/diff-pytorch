import torch
from typing import Tuple
import math
from dataclasses import dataclass

def linear_diffusion_schedule(diffusion_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear diffusion schedule.

    Args:
        diffusion_times (torch.Tensor): Diffusion times of size (batch_size, 1, 1, 1)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Noise rates and signal rates of size (batch_size, 1, 1, 1) each.
    """    
    # Check shape
    assert diffusion_times.shape == torch.Size([diffusion_times.shape[0], 1, 1, 1]), f"Expected shape (batch_size, 1, 1, 1), got {diffusion_times.shape}"

    # Min rate
    min_rate = 0.0001
    max_rate = 0.02
    
    # Compuute betas and alphas
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas

    # Alpha bar = prod
    alpha_bars = torch.cumprod(alphas, dim=0)

    # Signal = sqrt
    signal_rates = torch.sqrt(alpha_bars)

    # Noise rate
    noise_rates = torch.sqrt(1 - alpha_bars)
    return noise_rates, signal_rates

def cosine_diffusion_schedule(diffusion_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine diffusion schedule.

    Args:
        diffusion_times (torch.Tensor): Diffusion times of size (batch_size, 1, 1, 1)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Noise rates and signal rates of size (batch_size, 1, 1, 1) each.
    """    
    # Check shape
    assert diffusion_times.shape == torch.Size([diffusion_times.shape[0], 1, 1, 1]), f"Expected shape (batch_size, 1, 1, 1), got {diffusion_times.shape}"

    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates

def offset_cosine_diffusion_schedule(diffusion_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Offset cosine diffusion schedule.

    Args:
        diffusion_times (torch.Tensor): Diffusion times of size (batch_size, 1, 1, 1)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Noise rates and signal rates of size (batch_size, 1, 1, 1) each.
    """    
    # Check shape
    assert diffusion_times.shape == torch.Size([diffusion_times.shape[0], 1, 1, 1]), f"Expected shape (batch_size, 1, 1, 1), got {diffusion_times.shape}"
    
    min_signal_rate = torch.tensor(0.02)
    max_signal_rate = torch.tensor(0.95)
    start_angle = torch.acos(max_signal_rate)
    end_angle = torch.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates

@dataclass
class DiffusionSchedules:
    T: int
    "Number of diffusion steps"

    diffusion_times: torch.Tensor
    "Diffusion times of shape (T)"

    linear_noise_rates: torch.Tensor
    "Linear noise rates of shape (T)"

    linear_signal_rates: torch.Tensor
    "Linear signal rates of shape (T)"

    cosine_noise_rates: torch.Tensor
    "Cosine noise rates of shape (T)"

    cosine_signal_rates: torch.Tensor
    "Cosine signal rates of shape (T)"

    offset_cosine_noise_rates: torch.Tensor
    "Offset cosine noise rates of shape (T)"

    offset_cosine_signal_rates: torch.Tensor
    "Offset cosine signal rates of shape (T)"

def create_all_diffusion_schedules(T: int) -> DiffusionSchedules:
    """Creates all diffusion schedules.

    Args:
        T (int): Number of diffusion steps.

    Returns:
        DiffusionSchedules: Diffusion schedules.
    """    
    diffusion_times = torch.tensor([x / T for x in range(T)]) # shape = [T]
    lnr, lsr = linear_diffusion_schedule(diffusion_times) # shape = [T]
    cnr, csr = cosine_diffusion_schedule(diffusion_times) # shape = [T]
    ocnr, ocsr = offset_cosine_diffusion_schedule(diffusion_times) # shape = [T]
    return DiffusionSchedules(
        T=T,
        diffusion_times=diffusion_times,
        linear_noise_rates=lnr,
        linear_signal_rates=lsr,
        cosine_noise_rates=cnr,
        cosine_signal_rates=csr,
        offset_cosine_noise_rates=ocnr,
        offset_cosine_signal_rates=ocsr,
        )

def plot_diffusion_schedules(ds: DiffusionSchedules):
    """Plots diffusion schedules.

    Args:
        ds (DiffusionSchedules): Diffusion schedules.
    """    
    import plotly.graph_objs as go

    # Create a figure
    fig1 = go.Figure()

    # Add traces for the first set of data
    fig1.add_trace(go.Scatter(
        x=ds.diffusion_times,
        y=ds.linear_signal_rates**2,
        mode='lines',
        name='linear'
    ))
    fig1.add_trace(go.Scatter(
        x=ds.diffusion_times,
        y=ds.cosine_signal_rates**2,
        mode='lines',
        name='cosine'
    ))
    fig1.add_trace(go.Scatter(
        x=ds.diffusion_times,
        y=ds.offset_cosine_signal_rates**2,
        mode='lines',
        name='offset_cosine'
    ))

    # Customize layout
    fig1.update_xaxes(title_text="t/T", tickfont=dict(size=12))
    fig1.update_yaxes(title_text=r"$\bar{\alpha_t}$ (signal)", tickfont=dict(size=12))
    fig1.update_layout(legend=dict(title=dict(text='Legend')))

    # Show the plot
    fig1.show()

    # Create a new figure for the second set of data
    fig2 = go.Figure()

    # Add traces for the second set of data
    fig2.add_trace(go.Scatter(
        x=ds.diffusion_times,
        y=ds.linear_noise_rates**2,
        mode='lines',
        name='linear'
    ))
    fig2.add_trace(go.Scatter(
        x=ds.diffusion_times,
        y=ds.cosine_noise_rates**2,
        mode='lines',
        name='cosine'
    ))
    fig2.add_trace(go.Scatter(
        x=ds.diffusion_times,
        y=ds.offset_cosine_noise_rates**2,
        mode='lines',
        name='offset_cosine'
    ))

    # Customize layout for the second plot
    fig2.update_xaxes(title_text="t/T", tickfont=dict(size=12))
    fig2.update_yaxes(title_text=r"$1-\bar{\alpha_t}$ (noise)", tickfont=dict(size=12))
    fig2.update_layout(legend=dict(title=dict(text='Legend')))

    # Show the second plot
    fig2.show()
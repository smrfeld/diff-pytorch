import torch
from typing import Tuple
import math
from dataclasses import dataclass

def linear_diffusion_schedule(diffusion_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates

def offset_cosine_diffusion_schedule(diffusion_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    diffusion_times: torch.Tensor
    linear_noise_rates: torch.Tensor
    linear_signal_rates: torch.Tensor
    cosine_noise_rates: torch.Tensor
    cosine_signal_rates: torch.Tensor
    offset_cosine_noise_rates: torch.Tensor
    offset_cosine_signal_rates: torch.Tensor

def create_all_diffusion_schedules(T: int) -> DiffusionSchedules:
    diffusion_times = torch.tensor([x / T for x in range(T)])
    lnr, lsr = linear_diffusion_schedule(diffusion_times)
    cnr, csr = cosine_diffusion_schedule(diffusion_times)
    ocnr, ocsr = offset_cosine_diffusion_schedule(diffusion_times)
    return DiffusionSchedules(
        diffusion_times=diffusion_times,
        linear_noise_rates=lnr,
        linear_signal_rates=lsr,
        cosine_noise_rates=cnr,
        cosine_signal_rates=csr,
        offset_cosine_noise_rates=ocnr,
        offset_cosine_signal_rates=ocsr,
        )

def plot_diffusion_schedules(ds: DiffusionSchedules):
    import matplotlib.pyplot as plt

    plt.plot(
        ds.diffusion_times, ds.linear_signal_rates**2, linewidth=1.5, label="linear"
    )
    plt.plot(
        ds.diffusion_times, ds.cosine_signal_rates**2, linewidth=1.5, label="cosine"
    )
    plt.plot(
        ds.diffusion_times,
        ds.offset_cosine_signal_rates**2,
        linewidth=1.5,
        label="offset_cosine",
    )

    plt.xlabel("t/T", fontsize=12)
    plt.ylabel(r"$\bar{\alpha_t}$ (signal)", fontsize=12)
    plt.legend()
    plt.show()

    plt.plot(
        ds.diffusion_times, ds.linear_noise_rates**2, linewidth=1.5, label="linear"
    )
    plt.plot(
        ds.diffusion_times, ds.cosine_noise_rates**2, linewidth=1.5, label="cosine"
    )
    plt.plot(
        ds.diffusion_times,
        ds.offset_cosine_noise_rates**2,
        linewidth=1.5,
        label="offset_cosine",
    )

    plt.xlabel("t/T", fontsize=12)
    plt.ylabel(r"$1-\bar{\alpha_t}$ (noise)", fontsize=12)
    plt.legend()
    plt.show()

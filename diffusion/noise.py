# nb_sim/diffusion/noise.py
from __future__ import annotations
import torch
import math

def alpha_bar_cosine(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    # ᾱ(t) = cos^2(π/2 * (t+s)/(1+s))
    t = t.clamp(0.0, 1.0)
    f = (t + s) / (1.0 + s)
    return torch.cos(0.5 * math.pi * f).pow(2)

def alpha_sigma_from_t(t: torch.Tensor, s: float = 0.008):
    abar = alpha_bar_cosine(t, s=s)
    return abar.sqrt(), (1.0 - abar).sqrt()

def alpha_sigma_from_u(u: torch.Tensor):
    abar = torch.sigmoid(u)
    return abar.sqrt(), (1.0 - abar).sqrt()

def s_over_a_from_u(u: torch.Tensor):
    # σ/α = exp(-u/2)
    return torch.exp(-0.5 * u)

def add_noise(z0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    alpha_t, sigma_t = alpha_sigma_from_t(t)
    while alpha_t.dim() < z0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
    return alpha_t * z0 + sigma_t * eps

def logsnr(t: torch.Tensor):
    a, s = alpha_sigma_from_t(t)
    return torch.log((a*a) / (s*s + 1e-12) + 1e-20)

# ---- Forward: t -> logSNR ----
def logsnr_from_t(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    abar = alpha_bar_cosine(t, s=s).clamp(1e-12, 1.0 - 1e-12)
    # u = log(ᾱ/(1-ᾱ)) computed stably
    return torch.log(abar) - torch.log1p(-abar)

# ---- Inverse: logSNR -> t ----
def t_from_logsnr(u: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    # ᾱ = sigmoid(u)  (since u = log(ᾱ/(1-ᾱ)))
    abar = torch.sigmoid(u).clamp(1e-12, 1.0 - 1e-12)
    # Invert ᾱ = cos^2(π/2 * (t+s)/(1+s)):
    # (t+s)/(1+s) = (2/π) * acos(sqrt(ᾱ))
    f = (2.0 / math.pi) * torch.acos(abar.sqrt())
    t = f * (1.0 + s) - s
    return t.clamp(0.0, 1.0)
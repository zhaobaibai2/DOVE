import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BetaScheduleCoefficients:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    log_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor


def _vp_beta_schedule(timesteps: int) -> np.ndarray:
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / (T ** 2))
    betas = 1.0 - alpha
    return betas.astype(np.float32)


def _precompute_coefficients(num_timesteps: int, *, device: torch.device) -> BetaScheduleCoefficients:
    betas = _vp_beta_schedule(num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1.0)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    return BetaScheduleCoefficients(
        betas=t(betas),
        alphas=t(alphas),
        alphas_cumprod=t(alphas_cumprod),
        alphas_cumprod_prev=t(alphas_cumprod_prev),
        sqrt_alphas_cumprod=t(sqrt_alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=t(sqrt_one_minus_alphas_cumprod),
        log_one_minus_alphas_cumprod=t(log_one_minus_alphas_cumprod),
        sqrt_recip_alphas_cumprod=t(sqrt_recip_alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=t(sqrt_recipm1_alphas_cumprod),
        posterior_variance=t(posterior_variance),
        posterior_log_variance_clipped=t(posterior_log_variance_clipped),
        posterior_mean_coef1=t(posterior_mean_coef1),
        posterior_mean_coef2=t(posterior_mean_coef2),
    )


class GaussianDiffusion:
    def __init__(self, num_timesteps: int = 20, *, device: torch.device):
        assert num_timesteps > 0
        self.num_timesteps = int(num_timesteps)
        self.device = device
        self.B = _precompute_coefficients(self.num_timesteps, device=device)

    def predict_x0_from_noise(self, t_index: int, x: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        B = self.B
        x_recon = x * B.sqrt_recip_alphas_cumprod[t_index] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t_index]
        return x_recon.clamp(-1.0, 1.0)

    def p_mean_variance(self, t_index: int, x: torch.Tensor, noise_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.B
        x_recon = self.predict_x0_from_noise(t_index, x, noise_pred)
        model_mean = x_recon * B.posterior_mean_coef1[t_index] + x * B.posterior_mean_coef2[t_index]
        model_log_variance = B.posterior_log_variance_clipped[t_index]
        return model_mean, model_log_variance

    def p_sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        *,
        return_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        x = torch.randn(shape, device=self.device, dtype=torch.float32)
        noise = torch.randn((self.num_timesteps, *shape), device=self.device, dtype=torch.float32)

        for t in reversed(range(self.num_timesteps)):
            t_vec = torch.full(shape[:-1], t, device=self.device, dtype=torch.long)
            noise_pred = model(t_vec, x)
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            if t > 0:
                x = model_mean + torch.exp(0.5 * model_log_variance) * noise[t] * 0.1
            else:
                x = model_mean
        if return_info:
            return x, {"diffusion/steps_used": float(self.num_timesteps), "diffusion/sampler": 0.0}
        return x

    def p_sample_deterministic(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        *,
        return_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        x = torch.zeros(shape, device=self.device, dtype=torch.float32)
        for t in reversed(range(self.num_timesteps)):
            t_vec = torch.full(shape[:-1], t, device=self.device, dtype=torch.long)
            noise_pred = model(t_vec, x)
            model_mean, _ = self.p_mean_variance(t, x, noise_pred)
            x = model_mean
        if return_info:
            return x, {"diffusion/steps_used": float(self.num_timesteps), "diffusion/sampler": 1.0}
        return x

    def p_sample_ddim(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        *,
        ddim_steps: Optional[int] = None,
        eta: float = 0.0,
        start_from_noise: bool = True,
        guidance_fn: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, Dict[str, float]]]] = None,
        return_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        """DDIM sampler for low-latency vehicle inference.

        guidance_fn receives the Tweedie clean-action estimate x0 and returns a
        possibly guided x0 plus scalar diagnostics. It is intentionally optional
        so Phase2 BC and unguided baselines use the same sampler code path.
        """
        steps = int(ddim_steps or self.num_timesteps)
        steps = max(1, min(steps, self.num_timesteps))
        times = torch.linspace(self.num_timesteps - 1, 0, steps, device=self.device)
        times = torch.round(times).long().unique_consecutive()
        if int(times[-1].item()) != 0:
            times = torch.cat([times, torch.zeros(1, device=self.device, dtype=torch.long)])

        if start_from_noise:
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
        else:
            x = torch.zeros(shape, device=self.device, dtype=torch.float32)

        info_acc: Dict[str, float] = {
            "diffusion/steps_used": float(len(times)),
            "diffusion/sampler": 2.0,
            "guidance/applied_steps": 0.0,
        }

        for i, t_tensor in enumerate(times):
            t = int(t_tensor.item())
            t_vec = torch.full(shape[:-1], t, device=self.device, dtype=torch.long)
            eps = model(t_vec, x)
            x0 = self.predict_x0_from_noise(t, x, eps)

            if guidance_fn is not None:
                x0, step_info = guidance_fn(x0, t)
                x0 = x0.clamp(-1.0, 1.0)
                info_acc["guidance/applied_steps"] += float(step_info.get("guidance/applied", 0.0))
                for k, v in step_info.items():
                    try:
                        info_acc[k] = float(v)
                    except Exception:
                        pass

            if i == len(times) - 1:
                x = x0
                break

            t_prev = int(times[i + 1].item())
            alpha_t = self.B.alphas_cumprod[t]
            alpha_prev = self.B.alphas_cumprod_prev[t] if t_prev < 0 else self.B.alphas_cumprod[t_prev]
            sigma = float(eta) * torch.sqrt(
                torch.clamp((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev), min=0.0)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma ** 2, min=0.0)) * eps
            noise = torch.randn_like(x) if float(eta) > 0.0 and t_prev > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x0 + dir_xt + sigma * noise

        if return_info:
            info_acc["guidance/applied_ratio"] = info_acc["guidance/applied_steps"] / max(1.0, info_acc["diffusion/steps_used"])
            return x, info_acc
        return x

    def q_sample(self, t: torch.Tensor, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = self.B
        t = t.long()
        c1 = B.sqrt_alphas_cumprod[t].unsqueeze(-1)
        c2 = B.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return c1 * x_start + c2 * noise

    def p_loss(self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        return F.mse_loss(noise_pred, noise)

    def weighted_p_loss(self, weights: torch.Tensor, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        if weights.ndim == 1:
            weights = weights.view(-1, 1)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(t, x_start, noise)
        noise_pred = model(t, x_noisy)

        per_elem = (noise_pred - noise) ** 2
        per_sample = per_elem.mean(dim=-1)
        w = weights.squeeze(-1)
        return (per_sample * w).sum() / (w.sum() + 1e-6)

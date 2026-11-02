import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_diffusion import GaussianDiffusion


def _activation(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


def scaled_sinusoidal_encoding(t: torch.Tensor, *, dim: int, theta: int = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    t = t.float()
    half_dim = dim // 2
    emb = math.log(theta)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * (-emb / half_dim))
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, *, activation: str = "relu", use_layer_norm: bool = True):
        super().__init__()
        self.act = _activation(activation)
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU() if activation.lower() == "relu" else nn.GELU() if activation.lower() == "gelu" else nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Sequence[int], *, activation: str = "relu", use_layer_norm: bool = True):
        super().__init__()
        self.backbone = MLP(obs_dim + act_dim, hidden_dims, 2, activation=activation, use_layer_norm=use_layer_norm)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        x = torch.cat([obs, act], dim=-1)
        out = self.backbone(x)
        q_mean = out[..., 0]
        q_std = F.softplus(out[..., 1]) + 0.1
        return q_mean, q_std


class DACERPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Sequence[int], *, time_dim: int = 16, activation: str = "relu", use_layer_norm: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.time_dim = time_dim
        self.act_fn = _activation(activation)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU() if activation.lower() == "relu" else nn.GELU() if activation.lower() == "gelu" else nn.Tanh(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.mlp = MLP(obs_dim + act_dim + time_dim, hidden_dims, act_dim, activation=activation, use_layer_norm=use_layer_norm)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        if t.dim() == 0:
            t = t.view(1)
        if t.dim() == 1 and t.shape[0] == 1 and obs.shape[0] > 1:
            t = t.expand(obs.shape[0])

        te = scaled_sinusoidal_encoding(t, dim=self.time_dim)
        te = self.time_mlp(te)
        x = torch.cat([obs, act, te], dim=-1)
        return self.mlp(x)


@dataclass
class DACERActionConfig:
    init_alpha: float = 0.1
    action_noise_scale: float = 0.05
    use_ddim: bool = True
    ddim_steps: int = 5
    ddim_eta: float = 0.0
    use_guidance: bool = True
    guidance_scale: float = 0.04
    guidance_uncertainty_kappa: float = 1.0
    guidance_min_gate: float = 0.05
    guidance_sigma_ref: float = 1.0
    guidance_grad_clip: float = 1.0
    final_gate_enabled: bool = True
    final_gate_max_shift: float = 0.45
    final_gate_min_q_improve: float = 0.0
    final_gate_max_uncertainty: float = 0.50
    final_gate_use_uncertainty: bool = True


class DACERTorchAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        diffusion_hidden_dims: Sequence[int],
        *,
        num_timesteps: int = 20,
        target_entropy: float = -2.0,
        time_dim: int = 16,
        activation: str = "relu",
        use_layer_norm: bool = True,
        action_cfg: Optional[DACERActionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_timesteps = int(num_timesteps)
        self.target_entropy = float(target_entropy)
        self.device = device or torch.device("cpu")

        self.q1 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.q2 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.target_q1 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.target_q2 = QNetwork(self.obs_dim, self.act_dim, hidden_dims, activation=activation, use_layer_norm=use_layer_norm)
        self.policy_net = DACERPolicyNet(self.obs_dim, self.act_dim, diffusion_hidden_dims, time_dim=time_dim, activation=activation, use_layer_norm=use_layer_norm)

        self.diffusion = GaussianDiffusion(self.num_timesteps, device=self.device)

        action_cfg = action_cfg or DACERActionConfig()
        self.action_noise_scale = float(action_cfg.action_noise_scale)
        self.use_ddim = bool(action_cfg.use_ddim)
        self.ddim_steps = int(action_cfg.ddim_steps)
        self.ddim_eta = float(action_cfg.ddim_eta)
        self.use_guidance = bool(action_cfg.use_guidance)
        self.guidance_scale = float(action_cfg.guidance_scale)
        self.guidance_uncertainty_kappa = float(action_cfg.guidance_uncertainty_kappa)
        self.guidance_min_gate = float(action_cfg.guidance_min_gate)
        self.guidance_sigma_ref = float(action_cfg.guidance_sigma_ref)
        self.guidance_grad_clip = float(action_cfg.guidance_grad_clip)
        self.final_gate_enabled = bool(action_cfg.final_gate_enabled)
        self.final_gate_max_shift = float(action_cfg.final_gate_max_shift)
        self.final_gate_min_q_improve = float(action_cfg.final_gate_min_q_improve)
        self.final_gate_max_uncertainty = float(action_cfg.final_gate_max_uncertainty)
        self.final_gate_use_uncertainty = bool(action_cfg.final_gate_use_uncertainty)
        init_alpha = float(action_cfg.init_alpha)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), dtype=torch.float32, device=self.device))

        self._sync_targets(tau=1.0)

    @torch.no_grad()
    def _sync_targets(self, tau: float = 1.0):
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    @torch.no_grad()
    def soft_update_targets(self, tau: float):
        self._sync_targets(tau=tau)

    def cpsi(self, obs: torch.Tensor, act: torch.Tensor, *, target: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1_mean, _ = self.q(1, obs, act, target=target)
        c2_mean, _ = self.q(2, obs, act, target=target)
        c_lcb = torch.minimum(c1_mean, c2_mean)
        c_ucb = torch.maximum(c1_mean, c2_mean)
        c_mean = 0.5 * (c1_mean + c2_mean)
        u_psi = torch.abs(c1_mean - c2_mean)
        return c_ucb, c_lcb, c_mean, u_psi

    def proxy_value(self, obs: torch.Tensor, act: torch.Tensor, *, target: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cpsi(obs, act, target=target)

    def _sample_once(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool,
        use_ddim: bool,
        use_guidance: bool,
        return_info: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        obs = obs.to(self.device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        def model(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.policy_net(obs, x, t)

        def guidance_fn(x0: torch.Tensor, t_index: int):
            if not bool(use_guidance) or float(self.guidance_scale) <= 0.0:
                return x0, {"guidance/applied": 0.0} if return_info else {}

            x_leaf = x0.detach().clone().requires_grad_(True)
            c_ucb, c_lcb, c_mean, u_psi = self.cpsi(obs, x_leaf, target=True)
            c_scalar = c_ucb.sum()
            grad = torch.autograd.grad(c_scalar, x_leaf, retain_graph=False, create_graph=False, allow_unused=True)[0]
            if grad is None:
                return x0, {"guidance/applied": 0.0} if return_info else {}

            grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
            grad_norm_raw = grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            clip_scale = torch.clamp(float(self.guidance_grad_clip) / grad_norm_raw, max=1.0)
            grad = grad * clip_scale
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            sigma_ref = max(float(self.guidance_sigma_ref), 1e-6)
            unc_gate = torch.exp(-float(self.guidance_uncertainty_kappa) * u_psi / sigma_ref).view(-1, 1)
            align_gate = torch.ones_like(unc_gate)
            gate = unc_gate.clamp(0.0, 1.0)
            gate = torch.where(gate >= float(self.guidance_min_gate), gate, torch.zeros_like(gate))
            guided = x0 - float(self.guidance_scale) * gate * grad / grad_norm

            if not return_info:
                return guided, {}

            return guided, {
                "guidance/applied": float((gate > 0.0).float().mean().detach().cpu()),
                "guidance/lambda": float((float(self.guidance_scale) * gate).mean().detach().cpu()),
                "guidance/gate": float(gate.mean().detach().cpu()),
                "guidance/uncertainty_gate": float(unc_gate.mean().detach().cpu()),
                "guidance/alignment_gate": float(align_gate.mean().detach().cpu()),
                "guidance/q_lcb": float(c_lcb.mean().detach().cpu()),
                "guidance/q_mean": float(c_ucb.mean().detach().cpu()),
                "guidance/proxy_score_mean": float(c_ucb.mean().detach().cpu()),
                "guidance/q_uncertainty": float(u_psi.mean().detach().cpu()),
                "guidance/sigma_q": float(u_psi.mean().detach().cpu()),
                "guidance/sigma_ref": float(sigma_ref),
                "guidance/grad_norm": float(grad_norm.mean().detach().cpu()),
                "guidance/q_grad_norm": float(grad_norm.mean().detach().cpu()),
                "guidance/grad_clip_frac": float((clip_scale < 1.0).float().mean().detach().cpu()),
                "guidance/timestep": float(t_index),
                "guidance/objective_is_cost": 1.0,
            }

        if bool(use_ddim):
            return self.diffusion.p_sample_ddim(
                model,
                (obs.shape[0], self.act_dim),
                ddim_steps=self.ddim_steps,
                eta=self.ddim_eta if not deterministic else 0.0,
                start_from_noise=not deterministic,
                guidance_fn=guidance_fn if use_guidance else None,
                return_info=return_info,
            )

        if deterministic:
            return self.diffusion.p_sample_deterministic(model, (obs.shape[0], self.act_dim), return_info=return_info)
        return self.diffusion.p_sample(model, (obs.shape[0], self.act_dim), return_info=return_info)

    def get_action(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        add_noise: bool = True,
        use_ddim: Optional[bool] = None,
        use_guidance: Optional[bool] = None,
        return_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        obs = obs.to(self.device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        use_ddim = self.use_ddim if use_ddim is None else bool(use_ddim)
        use_guidance = self.use_guidance if use_guidance is None else bool(use_guidance)
        sample = self._sample_once(
            obs,
            deterministic=deterministic,
            use_ddim=use_ddim,
            use_guidance=use_guidance,
            return_info=return_info,
        )
        if return_info:
            action, info = sample
        else:
            action = sample
            info = {}

        if bool(add_noise):
            noise = torch.randn_like(action)
            action = action + noise * torch.exp(self.log_alpha) * self.action_noise_scale
        action = action.clamp(-1.0, 1.0)
        if return_info:
            return action, info
        return action

    def get_action_with_gate_debug(self, obs: torch.Tensor, *, deterministic: bool = True, add_noise: bool = False) -> Tuple[torch.Tensor, dict]:
        obs = obs.to(self.device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            unguided, info_u = self.get_action(
                obs,
                deterministic=deterministic,
                add_noise=False,
                use_ddim=self.use_ddim,
                use_guidance=False,
                return_info=True,
            )

        guided, info_g = self.get_action(
            obs,
            deterministic=deterministic,
            add_noise=add_noise,
            use_ddim=self.use_ddim,
            use_guidance=self.use_guidance,
            return_info=True,
        )

        c_u, c_lcb_u, _, unc_u = self.cpsi(obs, unguided, target=True)
        c_g, c_lcb_g, _, unc_g = self.cpsi(obs, guided, target=True)
        shift = torch.norm(guided - unguided, dim=-1)
        proxy_gain = (c_u - c_g).view(-1)
        unc_g_1d = unc_g.view(-1)
        accept = torch.ones_like(shift, dtype=torch.bool)
        shift_ok = shift <= float(self.final_gate_max_shift)
        proxy_ok = proxy_gain >= float(self.final_gate_min_q_improve)
        uncertainty_ok = unc_g_1d <= float(self.final_gate_max_uncertainty)
        if bool(self.final_gate_enabled):
            accept = shift_ok & proxy_ok
            if bool(self.final_gate_use_uncertainty):
                accept = accept & uncertainty_ok

        action = torch.where(accept.view(-1, 1), guided, unguided).clamp(-1.0, 1.0)
        info = dict(info_u)
        info.update(info_g)
        info.update({
            "gate/accepted": float(accept.float().mean().detach().cpu()),
            "gate/fallback": float((~accept).float().mean().detach().cpu()),
            "gate/action_shift_l2": float(shift.mean().detach().cpu()),
            "gate/q_lcb_unguided": float(c_lcb_u.mean().detach().cpu()),
            "gate/q_lcb_guided": float(c_lcb_g.mean().detach().cpu()),
            "gate/q_lcb_improve": float(proxy_gain.mean().detach().cpu()),
            "gate/proxy_gain": float(proxy_gain.mean().detach().cpu()),
            "gate/delta_c": float((c_g - c_u).mean().detach().cpu()),
            "gate/q_unguided": float(c_u.mean().detach().cpu()),
            "gate/q_guided": float(c_g.mean().detach().cpu()),
            "gate/unc_unguided": float(unc_u.mean().detach().cpu()),
            "gate/unc_guided": float(unc_g.mean().detach().cpu()),
            "gate/proxy_ok": float(proxy_ok.float().mean().detach().cpu()),
            "gate/uncertainty_ok": float(uncertainty_ok.float().mean().detach().cpu()),
            "gate/uncertainty_gate_enabled": float(1.0 if self.final_gate_use_uncertainty else 0.0),
            "gate/shift_ok": float(shift_ok.float().mean().detach().cpu()),
        })
        return action, info

    def get_action_with_gate_fast(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = True,
        add_noise: bool = False,
        log_info: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        obs = obs.to(self.device, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            unguided = self.get_action(
                obs,
                deterministic=deterministic,
                add_noise=False,
                use_ddim=self.use_ddim,
                use_guidance=False,
                return_info=False,
            )

        guided = self.get_action(
            obs,
            deterministic=deterministic,
            add_noise=add_noise,
            use_ddim=self.use_ddim,
            use_guidance=self.use_guidance,
            return_info=False,
        )

        with torch.no_grad():
            obs2 = obs.repeat(2, 1)
            act2 = torch.cat([unguided, guided], dim=0)
            c_ucb, c_lcb, _, unc = self.cpsi(obs2, act2, target=True)

            c_u = c_ucb[0:1]
            c_g = c_ucb[1:2]
            c_lcb_u = c_lcb[0:1]
            c_lcb_g = c_lcb[1:2]
            unc_u = unc[0:1]
            unc_g = unc[1:2]

            shift = torch.norm(guided - unguided, dim=-1)
            proxy_gain = (c_u - c_g).view(-1)
            unc_g_1d = unc_g.view(-1)
            shift_ok = shift <= float(self.final_gate_max_shift)
            proxy_ok = proxy_gain >= float(self.final_gate_min_q_improve)
            uncertainty_ok = unc_g_1d <= float(self.final_gate_max_uncertainty)

            if bool(self.final_gate_enabled):
                accept = shift_ok & proxy_ok
                if bool(self.final_gate_use_uncertainty):
                    accept = accept & uncertainty_ok
            else:
                accept = torch.ones_like(shift, dtype=torch.bool)

            action = torch.where(accept.view(-1, 1), guided, unguided).clamp(-1.0, 1.0)

            if not log_info:
                return action, {}

            info = {
                "gate/accepted": float(accept.float().mean().detach().cpu()),
                "gate/fallback": float((~accept).float().mean().detach().cpu()),
                "gate/action_shift_l2": float(shift.mean().detach().cpu()),
                "gate/q_lcb_unguided": float(c_lcb_u.mean().detach().cpu()),
                "gate/q_lcb_guided": float(c_lcb_g.mean().detach().cpu()),
                "gate/q_lcb_improve": float(proxy_gain.mean().detach().cpu()),
                "gate/proxy_gain": float(proxy_gain.mean().detach().cpu()),
                "gate/delta_c": float((c_g - c_u).mean().detach().cpu()),
                "gate/q_unguided": float(c_u.mean().detach().cpu()),
                "gate/q_guided": float(c_g.mean().detach().cpu()),
                "gate/unc_unguided": float(unc_u.mean().detach().cpu()),
                "gate/unc_guided": float(unc_g.mean().detach().cpu()),
                "gate/proxy_ok": float(proxy_ok.float().mean().detach().cpu()),
                "gate/uncertainty_ok": float(uncertainty_ok.float().mean().detach().cpu()),
                "gate/uncertainty_gate_enabled": float(1.0 if self.final_gate_use_uncertainty else 0.0),
                "gate/shift_ok": float(shift_ok.float().mean().detach().cpu()),
                "gate/diagnostics_logged": 1.0,
            }
            return action, info

    def get_action_with_gate(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = True,
        add_noise: bool = False,
        fast: bool = False,
        log_info: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        if fast:
            return self.get_action_with_gate_fast(
                obs,
                deterministic=deterministic,
                add_noise=add_noise,
                log_info=log_info,
            )
        return self.get_action_with_gate_debug(obs, deterministic=deterministic, add_noise=add_noise)

    def q(self, which: int, obs: torch.Tensor, act: torch.Tensor, *, target: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device, dtype=torch.float32)
        act = act.to(self.device, dtype=torch.float32)
        if target:
            net = self.target_q1 if which == 1 else self.target_q2
        else:
            net = self.q1 if which == 1 else self.q2
        return net(obs, act)

    def q_evaluate(self, which: int, obs: torch.Tensor, act: torch.Tensor, *, target: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_mean, q_std = self.q(which, obs, act, target=target)
        z = torch.randn_like(q_mean).clamp(-3.0, 3.0)
        q_val = q_mean + q_std * z
        return q_mean, q_std, q_val

    def predict_noise(self, obs: torch.Tensor, t: torch.Tensor, x_noisy: torch.Tensor) -> torch.Tensor:
        return self.policy_net(obs, x_noisy, t)

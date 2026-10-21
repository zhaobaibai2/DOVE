from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .torch_networks import DACERTorchAgent
from .torch_replay_buffer import PVPBatch


def compute_pv_loss(
    *,
    q_fn,
    obs: torch.Tensor,
    a_human: torch.Tensor,
    a_novice: torch.Tensor,
    interventions: torch.Tensor,
    B: float,
    critic_objective: str = "cost",
) -> torch.Tensor:
    I = interventions
    if I.dim() == 2 and I.shape[-1] == 1:
        I = I.squeeze(-1)

    q_h_mean, _ = q_fn(obs, a_human)
    q_n_mean, _ = q_fn(obs, a_novice)

    if q_h_mean.dim() > 1:
        q_h = q_h_mean.squeeze(-1)
    else:
        q_h = q_h_mean

    if q_n_mean.dim() > 1:
        q_n = q_n_mean.squeeze(-1)
    else:
        q_n = q_n_mean

    is_cost = str(critic_objective).lower() in ("cost", "risk", "lower", "lower_is_better")
    sign = -1.0 if is_cost else 1.0
    target_h = sign * float(B)
    target_n = -sign * float(B)
    pv = (q_h - target_h) ** 2 + (q_n - target_n) ** 2
    denom = I.sum() + 1e-6
    return (pv * I).sum() / denom


@dataclass
class TorchDACERConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 1e-4
    alpha_lr: float = 3e-2
    delay_update: int = 1
    delay_alpha_update: int = 1000
    reward_scale: float = 1.0
    lambda_pv: float = 1.0
    B: float = 1.0
    lambda_bc: float = 5.0
    reward_free: bool = True
    phase3_use_bc_boost: bool = True
    fix_alpha: bool = True
    target_entropy: float = -2.0
    lambda_energy: float = 0.5
    energy_margin: float = 0.5
    energy_min_action_gap: float = 0.03
    rejected_action_radius: float = 0.20
    critic_objective: str = "cost"


class PVPDACERTorch:
    def __init__(self, agent: DACERTorchAgent, cfg: TorchDACERConfig, *, device: torch.device):
        self.agent = agent
        self.cfg = cfg
        self.device = device

        self.optim_q1 = torch.optim.Adam(self.agent.q1.parameters(), lr=cfg.lr)
        self.optim_q2 = torch.optim.Adam(self.agent.q2.parameters(), lr=cfg.lr)
        self.optim_policy = torch.optim.Adam(self.agent.policy_net.parameters(), lr=cfg.lr)
        self.optim_alpha = torch.optim.Adam([self.agent.log_alpha], lr=cfg.alpha_lr)

        self.step = 0
        self.mean_q1_std = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
        self.mean_q2_std = torch.tensor(-1.0, device=self.device, dtype=torch.float32)
        self._last_entropy = torch.tensor(0.0, device=self.device, dtype=torch.float32)

    def get_action(self, obs: torch.Tensor, *, deterministic: bool = False, add_noise: bool = True) -> torch.Tensor:
        return self.agent.get_action(obs, deterministic=deterministic, add_noise=add_noise)

    def get_action_with_info(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = True,
        add_noise: bool = False,
        fast: bool = False,
        log_info: bool = True,
    ):
        return self.agent.get_action_with_gate(
            obs,
            deterministic=deterministic,
            add_noise=add_noise,
            fast=fast,
            log_info=log_info,
        )

    def _objective_twin(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        is_cost = str(self.cfg.critic_objective).lower() in ("cost", "risk", "lower", "lower_is_better")
        return torch.maximum(q1, q2) if is_cost else torch.minimum(q1, q2)

    def _update_mean_q_std(self, prev: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if prev.item() < 0.0:
            return new.detach()
        return (self.cfg.tau * new + (1.0 - self.cfg.tau) * prev).detach()

    def _q_loss_distributional(
        self,
        *,
        q_mean: torch.Tensor,
        q_std: torch.Tensor,
        backup_mean_1d: torch.Tensor,
        backup_sample_1d: torch.Tensor,
        mean_q_std: torch.Tensor,
        td_mask_1d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_mean_1d = q_mean.squeeze(-1) if q_mean.dim() > 1 else q_mean
        q_std_1d = q_std.squeeze(-1) if q_std.dim() > 1 else q_std

        q_std_detach = torch.maximum(q_std_1d.detach(), torch.zeros_like(q_std_1d))
        epsilon = 0.1

        q_backup_bounded = (q_mean_1d.detach() + (backup_sample_1d - q_mean_1d.detach()).clamp(-3.0 * mean_q_std, 3.0 * mean_q_std)).detach()

        per_sample_td = -((mean_q_std ** 2 + epsilon) * (
            q_mean_1d * (backup_mean_1d - q_mean_1d).detach() / (q_std_detach ** 2 + epsilon) +
            q_std_1d * (((q_mean_1d.detach() - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon))
        ))

        per_sample_td = per_sample_td * td_mask_1d
        return per_sample_td.mean(), q_mean_1d

    def train_offline_bc(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, float]:
        self.agent.train()
        obs = obs.to(self.device, dtype=torch.float32)
        action = action.to(self.device, dtype=torch.float32)

        t = torch.randint(0, self.agent.num_timesteps, (obs.shape[0],), device=self.device, dtype=torch.long)

        def model(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
            return self.agent.predict_noise(obs, t_batch, x_batch)

        loss = self.agent.diffusion.p_loss(model, t, action)
        loss = self.cfg.lambda_bc * loss

        self.optim_policy.zero_grad(set_to_none=True)
        loss.backward()
        self.optim_policy.step()

        return {'bc_loss': float(loss.detach().cpu())}

    def train_pvp(self, batch: PVPBatch) -> Dict[str, float]:
        self.agent.train()
        self.step += 1

        q_behavior_means = []
        q_novice_means = []

        obs = batch.obs
        next_obs = batch.next_obs
        reward = batch.reward * float(self.cfg.reward_scale)
        done = batch.done
        a_b = batch.actions_behavior
        a_n = batch.actions_novice
        a_h = batch.actions_human
        interventions = batch.interventions
        stop_td = batch.stop_td

        with torch.no_grad():
            next_action = self.agent.get_action(next_obs, deterministic=True, add_noise=False, use_guidance=False)
            next_q1_mean, next_q1_std, next_q1_val = self.agent.q_evaluate(1, next_obs, next_action, target=True)
            next_q2_mean, next_q2_std, next_q2_val = self.agent.q_evaluate(2, next_obs, next_action, target=True)
            is_cost = str(self.cfg.critic_objective).lower() in ("cost", "risk", "lower", "lower_is_better")
            next_q_mean = torch.maximum(next_q1_mean, next_q2_mean) if is_cost else torch.minimum(next_q1_mean, next_q2_mean)
            take_q1 = next_q1_mean > next_q2_mean if is_cost else next_q1_mean < next_q2_mean
            next_q_sample = torch.where(take_q1, next_q1_val, next_q2_val)

            q_target = next_q_mean
            q_target_sample = next_q_sample

            td_mask = (1.0 - stop_td).clamp(0.0, 1.0)

            if bool(self.cfg.reward_free):
                td_mask = td_mask * (1.0 - interventions)

            reward_1d = reward.squeeze(-1) if reward.dim() > 1 else reward
            done_1d = done.squeeze(-1) if done.dim() > 1 else done
            td_mask_1d = td_mask.squeeze(-1) if td_mask.dim() > 1 else td_mask

            q_target_1d = q_target.squeeze(-1) if q_target.dim() > 1 else q_target
            q_target_sample_1d = q_target_sample.squeeze(-1) if q_target_sample.dim() > 1 else q_target_sample

            td_reward_1d = torch.zeros_like(reward_1d) if bool(self.cfg.reward_free) else reward_1d
            if is_cost:
                td_reward_1d = -td_reward_1d
            backup_mean_1d = td_reward_1d + (1.0 - done_1d) * float(self.cfg.gamma) * q_target_1d
            backup_sample_1d = td_reward_1d + (1.0 - done_1d) * float(self.cfg.gamma) * q_target_sample_1d

        q1_mean, q1_std = self.agent.q(1, obs, a_b, target=False)
        q2_mean, q2_std = self.agent.q(2, obs, a_b, target=False)

        self.mean_q1_std = self._update_mean_q_std(self.mean_q1_std, q1_std.mean())
        self.mean_q2_std = self._update_mean_q_std(self.mean_q2_std, q2_std.mean())

        q1_td_loss, q1_mean_1d = self._q_loss_distributional(
            q_mean=q1_mean,
            q_std=q1_std,
            backup_mean_1d=backup_mean_1d,
            backup_sample_1d=backup_sample_1d,
            mean_q_std=self.mean_q1_std,
            td_mask_1d=td_mask_1d,
        )
        q2_td_loss, q2_mean_1d = self._q_loss_distributional(
            q_mean=q2_mean,
            q_std=q2_std,
            backup_mean_1d=backup_mean_1d,
            backup_sample_1d=backup_sample_1d,
            mean_q_std=self.mean_q2_std,
            td_mask_1d=td_mask_1d,
        )

        pv1 = compute_pv_loss(q_fn=lambda o, a: self.agent.q(1, o, a, target=False), obs=obs, a_human=a_h, a_novice=a_n, interventions=interventions, B=float(self.cfg.B), critic_objective=self.cfg.critic_objective)
        pv2 = compute_pv_loss(q_fn=lambda o, a: self.agent.q(2, o, a, target=False), obs=obs, a_human=a_h, a_novice=a_n, interventions=interventions, B=float(self.cfg.B), critic_objective=self.cfg.critic_objective)
        pv_loss = 0.5 * (pv1 + pv2)

        I = interventions.squeeze(-1) if interventions.dim() == 2 and interventions.shape[-1] == 1 else interventions
        with torch.no_grad():
            c1_pos, _ = self.agent.q(1, obs, a_h, target=True)
            c2_pos, _ = self.agent.q(2, obs, a_h, target=True)
            c1_neg, _ = self.agent.q(1, obs, a_n, target=True)
            c2_neg, _ = self.agent.q(2, obs, a_n, target=True)
            c_pos = self._objective_twin(c1_pos, c2_pos)
            c_neg = self._objective_twin(c1_neg, c2_neg)
            c_pos_1d = c_pos.squeeze(-1) if c_pos.dim() > 1 else c_pos
            c_neg_1d = c_neg.squeeze(-1) if c_neg.dim() > 1 else c_neg
            critic_gap = c_neg_1d - c_pos_1d if is_cost else c_pos_1d - c_neg_1d
            pair_sat_e = (((critic_gap > 0.0).float() * I).sum() / (I.sum() + 1e-6)).detach()
            iar = (((critic_gap <= 0.0).float() * I).sum() / (I.sum() + 1e-6)).detach()
            action_gap = torch.norm(a_h - a_n, dim=-1)

        q1_loss = q1_td_loss + float(self.cfg.lambda_pv) * pv1
        q2_loss = q2_td_loss + float(self.cfg.lambda_pv) * pv2

        self.optim_q1.zero_grad(set_to_none=True)
        q1_loss.backward(retain_graph=True)
        self.optim_q1.step()

        self.optim_q2.zero_grad(set_to_none=True)
        q2_loss.backward(retain_graph=True)
        self.optim_q2.step()

        policy_loss = torch.tensor(0.0, device=self.device)
        rl_loss = torch.tensor(0.0, device=self.device)
        bc_loss = torch.tensor(0.0, device=self.device)
        energy_loss = torch.tensor(0.0, device=self.device)
        energy_violation = torch.zeros_like(I)
        energy_rank_gap = torch.tensor(0.0, device=self.device)
        energy_rank_e_pos = torch.tensor(0.0, device=self.device)
        energy_rank_e_neg = torch.tensor(0.0, device=self.device)
        energy_rank_pair_ok_rate = torch.tensor(0.0, device=self.device)
        new_action = a_n.detach()

        q_behavior_means.append(float(q1_mean.mean().detach().cpu()))
        q_novice_means.append(float(q2_mean.mean().detach().cpu()))

        if self.step % int(self.cfg.delay_update) == 0:
            for p in self.agent.q1.parameters():
                p.requires_grad_(False)
            for p in self.agent.q2.parameters():
                p.requires_grad_(False)

            new_action = self.agent.get_action(obs, deterministic=True, add_noise=False, use_guidance=False)
            q1_pi, _ = self.agent.q(1, obs, new_action, target=False)
            q2_pi, _ = self.agent.q(2, obs, new_action, target=False)
            q_pi = self._objective_twin(q1_pi, q2_pi)

            I = interventions
            if I.dim() == 2 and I.shape[-1] == 1:
                I = I.squeeze(-1)
            notI = 1.0 - I
            q_scalar = q_pi.squeeze(-1) if q_pi.dim() > 1 else q_pi
            q1_beh, _ = self.agent.q(1, obs, a_b, target=False)
            q2_beh, _ = self.agent.q(2, obs, a_b, target=False)
            q_beh = self._objective_twin(q1_beh, q2_beh)
            q_beh_scalar = q_beh.squeeze(-1) if q_beh.dim() > 1 else q_beh
            improvement = (q_beh_scalar.detach() - q_scalar) if is_cost else (q_scalar - q_beh_scalar.detach())
            rl_loss = -(improvement * notI).sum() / (notI.sum() + 1e-6)

            if bool(self.cfg.phase3_use_bc_boost):
                weights = interventions
                t = torch.randint(0, self.agent.num_timesteps, (obs.shape[0],), device=self.device, dtype=torch.long)

                def model(t_batch: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
                    return self.agent.predict_noise(obs, t_batch, x_batch)

                bc_loss = self.agent.diffusion.weighted_p_loss(weights, model, t, a_h)
            else:
                bc_loss = torch.tensor(0.0, device=self.device)

            if float(self.cfg.lambda_energy) > 0.0:
                t_er = torch.randint(0, self.agent.num_timesteps, (obs.shape[0],), device=self.device, dtype=torch.long)
                shared_noise = torch.randn_like(a_h)
                x_pos_noisy = self.agent.diffusion.q_sample(t_er, a_h, shared_noise)
                x_neg_noisy = self.agent.diffusion.q_sample(t_er, a_n, shared_noise)
                pred_pos = self.agent.predict_noise(obs, t_er, x_pos_noisy)
                pred_neg = self.agent.predict_noise(obs, t_er, x_neg_noisy)
                e_pos = ((pred_pos - shared_noise) ** 2).mean(dim=-1)
                e_neg = ((pred_neg - shared_noise) ** 2).mean(dim=-1)
                pair_ok = (torch.norm(a_h - a_n, dim=-1) >= float(self.cfg.energy_min_action_gap)).float()
                er_w = I * pair_ok
                er_raw = float(self.cfg.energy_margin) + e_pos - e_neg
                energy_violation = F.relu(er_raw)
                energy_loss = (energy_violation * er_w).sum() / (er_w.sum() + 1e-6)
                energy_rank_gap = ((e_pos - e_neg) * er_w).sum() / (er_w.sum() + 1e-6)
                energy_rank_e_pos = (e_pos * er_w).sum() / (er_w.sum() + 1e-6)
                energy_rank_e_neg = (e_neg * er_w).sum() / (er_w.sum() + 1e-6)
                energy_rank_pair_ok_rate = (pair_ok * I).sum() / (I.sum() + 1e-6)

            policy_loss = rl_loss + float(self.cfg.lambda_bc) * bc_loss + float(self.cfg.lambda_energy) * energy_loss

            self.optim_policy.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.optim_policy.step()

            for p in self.agent.q1.parameters():
                p.requires_grad_(True)
            for p in self.agent.q2.parameters():
                p.requires_grad_(True)

            with torch.no_grad():
                self.agent.soft_update_targets(float(self.cfg.tau))

        alpha_loss = torch.tensor(0.0, device=self.device)
        if not bool(self.cfg.fix_alpha) and (self.step % int(self.cfg.delay_alpha_update) == 0) and self.step > 0:
            alpha = torch.exp(self.agent.log_alpha)
            alpha_loss = -(self.agent.log_alpha * (-self._last_entropy.detach() + float(self.cfg.target_entropy))).mean()
            self.optim_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.optim_alpha.step()

        with torch.no_grad():
            d_plus = torch.norm(new_action.detach() - a_h, dim=-1)
            d_minus = torch.norm(new_action.detach() - a_n, dim=-1)
            ragr = (((d_minus < float(self.cfg.rejected_action_radius)).float() * (1.0 - I)).sum() / ((1.0 - I).sum() + 1e-6))
            p_delta_plus = ((d_plus < float(self.cfg.rejected_action_radius)).float() * I).sum() / (I.sum() + 1e-6)
            p_delta_minus = ((d_minus < float(self.cfg.rejected_action_radius)).float() * I).sum() / (I.sum() + 1e-6)
            objective_gap = critic_gap
            risk_reduction_mean = improvement.detach().mean() if 'improvement' in locals() else torch.tensor(0.0, device=self.device)

        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "rl_loss": rl_loss.item(),
            "bc_loss": bc_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "energy_loss": energy_loss.item(),
            "energy_violation": energy_violation.mean().item(),
            "pair_sat_e": pair_sat_e.item(),
            "iar": iar.item(),
            "delta_e": objective_gap.mean().item(),
            "action_gap_l2": action_gap.mean().item(),
            "ragr": ragr.item(),
            "p_delta_plus": p_delta_plus.item(),
            "p_delta_minus": p_delta_minus.item(),
            "q1_mean": q1_mean.mean().item(),
            "q2_mean": q2_mean.mean().item(),
            "q_human": c_pos_1d.mean().item(),
            "q_rejected": c_neg_1d.mean().item(),
            "mean_q1_std": self.mean_q1_std.item(),
            "mean_q2_std": self.mean_q2_std.item(),
            "critic/objective": 1.0 if is_cost else 0.0,
            "critic/c_pos": c_pos_1d.mean().item(),
            "critic/c_neg": c_neg_1d.mean().item(),
            "critic/objective_gap": objective_gap.mean().item(),
            "policy/risk_reduction_nonexpert": risk_reduction_mean.item(),
            "policy/critic_objective": 1.0 if is_cost else 0.0,
            "energy_rank/loss": energy_loss.item(),
            "energy_rank/gap_pos_minus_neg": energy_rank_gap.item(),
            "energy_rank/gap_neg_minus_pos": (-energy_rank_gap).item(),
            "energy_rank/e_pos": energy_rank_e_pos.item(),
            "energy_rank/e_neg": energy_rank_e_neg.item(),
            "energy_rank/violation_rate": ((energy_violation > 0.0).float() * I).sum().item() / float(I.sum().item() + 1e-6),
            "energy_rank/pair_gap": action_gap.mean().item(),
            "energy_rank/pair_ok_rate": energy_rank_pair_ok_rate.item(),
            "energy_rank/margin": float(self.cfg.energy_margin),
        }

        
        return metrics

    def save(self, path: str) -> None:
        data = {
            'agent': self.agent.state_dict(),
            'optim_q1': self.optim_q1.state_dict(),
            'optim_q2': self.optim_q2.state_dict(),
            'optim_policy': self.optim_policy.state_dict(),
            'optim_alpha': self.optim_alpha.state_dict(),
            'step': self.step,
            'mean_q1_std': float(self.mean_q1_std.detach().cpu()),
            'mean_q2_std': float(self.mean_q2_std.detach().cpu()),
        }
        torch.save(data, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(data['agent'])
        self.optim_q1.load_state_dict(data.get('optim_q1', {}))
        self.optim_q2.load_state_dict(data.get('optim_q2', {}))
        self.optim_policy.load_state_dict(data.get('optim_policy', {}))
        self.optim_alpha.load_state_dict(data.get('optim_alpha', {}))
        self.step = int(data.get('step', 0))
        self.mean_q1_std = torch.tensor(float(data.get('mean_q1_std', -1.0)), device=self.device)
        self.mean_q2_std = torch.tensor(float(data.get('mean_q2_std', -1.0)), device=self.device)

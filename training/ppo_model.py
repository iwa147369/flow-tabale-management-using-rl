"""
Pointer-network actor-critic + PPO agent (Phase 3 architecture from docs/memory.md).

Design (permutation-invariant over the flow table):

    each flow [5 feat] ─┐
                        ├─ shared flow encoder (MLP 5→64→128) ─→ (N,128)
                        │        │
                        │        └─ maxpool over N ───────────→ global (128)
    incoming flow [5] ──── incoming encoder (MLP 5→64→128) ───→ (128)

    context = concat(global, incoming)                          (256)

    ACTOR  (pointer): score_i = MLP( concat(flow_enc_i, context) ) → logit_i
                       logits masked by eviction mask, softmax → policy over flows
    CRITIC          : V(s)    = MLP(context) → scalar

The actor is permutation-equivariant (every flow scored by the same head given a
shared, order-invariant context), so "which flow to evict" no longer depends on
table position — fixing GAP 3. Trained with PPO (no fixed action count, no
epsilon-greedy), per the redesign.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEG_INF = -1e9


def _mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class PointerActorCritic(nn.Module):
    def __init__(self, num_features=5, enc_dim=128, hidden=64):
        super().__init__()
        self.flow_encoder     = _mlp([num_features, hidden, enc_dim])
        self.incoming_encoder = _mlp([num_features, hidden, enc_dim])
        # actor scores each flow from its own encoding + the shared context
        self.actor_head = _mlp([enc_dim + 2 * enc_dim, hidden, 1])
        # critic values the state from the shared context only
        self.critic_head = _mlp([2 * enc_dim, hidden, 1])

    def _encode(self, table, incoming):
        """table: (B,N,F) incoming: (B,F) → flow_enc (B,N,E), context (B,2E)."""
        flow_enc = self.flow_encoder(table)               # (B,N,E)
        global_ctx = flow_enc.max(dim=1).values           # (B,E)  permutation-invariant
        inc_enc = self.incoming_encoder(incoming)         # (B,E)
        context = torch.cat([global_ctx, inc_enc], dim=-1)  # (B,2E)
        return flow_enc, context

    def forward(self, table, incoming, mask):
        flow_enc, context = self._encode(table, incoming)
        B, N, E = flow_enc.shape
        ctx_exp = context.unsqueeze(1).expand(B, N, context.shape[-1])  # (B,N,2E)
        actor_in = torch.cat([flow_enc, ctx_exp], dim=-1)               # (B,N,3E)
        logits = self.actor_head(actor_in).squeeze(-1)                  # (B,N)
        logits = logits.masked_fill(mask < 0.5, NEG_INF)
        value = self.critic_head(context).squeeze(-1)                   # (B,)
        return logits, value


def _to_batch(obs_list):
    """List of obs dicts → batched tensors on device."""
    table    = torch.as_tensor(np.stack([o["table"]    for o in obs_list]), dtype=torch.float32, device=device)
    incoming = torch.as_tensor(np.stack([o["incoming"] for o in obs_list]), dtype=torch.float32, device=device)
    mask     = torch.as_tensor(np.stack([o["mask"]     for o in obs_list]), dtype=torch.float32, device=device)
    return table, incoming, mask


class PPOAgent:
    def __init__(self, num_features=5, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=4, minibatch=256, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5):
        self.net = PointerActorCritic(num_features).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Single-obs action selection for rollout / inference."""
        table, incoming, mask = _to_batch([obs])
        logits, value = self.net(table, incoming, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def _gae(self, rewards, values, dones, last_value):
        adv = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            adv[t] = gae
        returns = adv + np.asarray(values, dtype=np.float32)
        return adv, returns

    def update(self, rollout, last_value):
        """rollout: dict of lists (obs, actions, logps, rewards, values, dones)."""
        adv, returns = self._gae(rollout["rewards"], rollout["values"],
                                 rollout["dones"], last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        table, incoming, mask = _to_batch(rollout["obs"])
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
        old_logps = torch.as_tensor(rollout["logps"], dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        n = len(actions)
        idx = np.arange(n)
        last_stats = {}
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch):
                b = idx[start:start + self.minibatch]
                logits, values = self.net(table[b], incoming[b], mask[b])
                dist = Categorical(logits=logits)
                logps = dist.log_prob(actions[b])
                ratio = torch.exp(logps - old_logps[b])

                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - ret_t[b]) ** 2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                last_stats = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
        return last_stats

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=device))

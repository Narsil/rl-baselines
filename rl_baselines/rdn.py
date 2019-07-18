from rl_baselines.core import logger, logdir, gae_advantages, discounted_returns
from rl_baselines.ppo import ppo_loss
from rl_baselines.model_updates import ValueUpdate
from torch.distributions import Categorical
from gym.spaces import Discrete, Box

import multiprocessing
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import psutil


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """

    assert len(y.shape) == 1 and len(ypred.shape) == 1
    vary = y.var()
    return np.nan if vary == 0 else 1 - (y - ypred).var() / vary


def ortho_init(tensor, scale=1.0):
    shape = tensor.shape
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4:  # assumes NHWC
        flat_shape = (np.prod(shape[:-1]), shape[-1])
    else:
        raise NotImplementedError
    a = torch.randn(flat_shape)
    u, _, v = torch.svd(a)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return (scale * q[: shape[0], : shape[1]]).float()


class RandomNet(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.int_convs = nn.ModuleList()
        layers = [
            {"out_channels": 32, "kernel_size": 8, "stride": 4},
            {"out_channels": 64, "kernel_size": 4, "stride": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 1},
        ]
        # We only look at last frame for surprise
        in_channels = 1
        for layer in layers:
            self.int_convs.append(nn.Conv2d(in_channels=in_channels, **layer))
            in_channels = layer["out_channels"]

        in_ = 3136
        self.fcs = nn.ModuleList()
        for out_ in hidden_sizes:
            self.fcs.append(nn.Linear(in_, out_))
            in_ = out_

        for conv in self.int_convs:
            nn.init.orthogonal_(conv.weight, gain=np.sqrt(2))
            conv.bias.data.zero_()

        for fc in self.fcs:
            nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
            fc.bias.data.zero_()

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, normed_obs):
        E, B, H, W, C = normed_obs.shape

        # We only look at last frame for surprise
        x = normed_obs.contiguous().view(-1, H, W, C)[:, :, :, -1:]
        # Pytorch uses C, H, W for its convolution
        x = x.permute(0, 3, 1, 2)

        for int_conv in self.int_convs:
            x = self.leaky_relu(int_conv(x))

        x = x.reshape(E * B, -1)

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i == len(self.fcs) - 1:
                x = self.relu(x)
        x = x.reshape(E, B, -1)
        return x


class CommonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        layers = [
            {"out_channels": 32, "kernel_size": 8, "stride": 4},
            {"out_channels": 64, "kernel_size": 4, "stride": 2},
            {"out_channels": 64, "kernel_size": 4, "stride": 1},
        ]
        in_channels = 4
        for layer in layers:
            self.convs.append(nn.Conv2d(in_channels=in_channels, **layer))
            in_channels = layer["out_channels"]

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 448)

        for layer in list(self.convs) + [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            layer.bias.data.zero_()

    def forward(self, obs):
        assert len(obs.shape) == 5
        E, B, H, W, C = obs.shape
        assert C == 4  # Frame stacks
        # Fuse environments and steps dimensions
        x = obs.contiguous().view(-1, H, W, C)
        # Pytorch uses C, H, W for its convolution
        x = x.permute(0, 3, 1, 2)
        for conv in self.convs:
            x = conv(x)
            x = self.relu(x)

        x = x.view(E * B, -1)
        x = self.relu(self.fc1(x))
        X = self.relu(self.fc2(x))
        return X


class GlobalModel(nn.Module):
    def __init__(self, n_acts):
        super().__init__()

        self.n_acts = n_acts

        self.common_model = CommonModel()

        self.fc_val = nn.Linear(448, 448)
        self.fc_act = nn.Linear(448, 448)
        self.fc_logits = nn.Linear(448, self.n_acts)
        self.fc_value_int = nn.Linear(448, 1)
        self.fc_value_ext = nn.Linear(448, 1)

        self.random_net = RandomNet([512])
        for p in self.random_net.parameters():
            p.requires_grad = False
        self.sibling_net = RandomNet([512, 512, 512])
        self.relu = nn.ReLU(inplace=True)

        for layer in [self.fc_val, self.fc_act]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(0.1))
            layer.bias.data.zero_()

        for layer in [self.fc_logits, self.fc_value_int, self.fc_value_ext]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(0.01))
            layer.bias.data.zero_()

    def forward(self, obs):
        batch_squeeze = False
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(1)
            batch_squeeze = True
            # Assume batch = 1
        assert len(obs.shape) == 5
        E, B, H, W, C = obs.shape
        assert C == 4  # Frame stacks
        X = self.common_model(obs)
        Xtout = X

        Xtout = X + self.relu(self.fc_val(Xtout))
        X = X + self.relu(self.fc_act(X))

        logits = self.fc_logits(X)
        value_int = self.fc_value_int(Xtout)
        value_ext = self.fc_value_ext(Xtout)

        if batch_squeeze:
            logits = logits.reshape(E, self.n_acts)
            value_int = value_int.reshape(E)
            value_ext = value_ext.reshape(E)
        else:
            logits = logits.reshape(E, B, self.n_acts)
            value_int = value_int.reshape(E, B)
            value_ext = value_ext.reshape(E, B)

        policy = Categorical(logits=logits)
        return policy, value_int, value_ext

    def policy(self, obs):
        return self.forward(obs)[0]

    def intrinsic_rewards(self, obs, obs_mean, obs_std):
        obs = (obs - obs_mean) / (obs_std + 1e-5)
        obs = torch.clamp(obs, -5.0, 5.0)
        with torch.no_grad():
            X_r = self.random_net(obs)
        X_r_hat = self.sibling_net(obs)

        rewards_loss = ((X_r - X_r_hat) ** 2).mean(dim=-1)
        return rewards_loss, X_r


class RDNValueUpdate(ValueUpdate):
    def __init__(
        self,
        model,
        optimizer,
        gamma,
        gamma_ext,
        lambda_,
        clip_ratio,
        iters,
        value_int_coeff,
        value_ext_coeff,
        ent_coeff,
        num_mini_batches,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.gamma_ext = gamma_ext
        self.lambda_ = lambda_
        self.clip_ratio = clip_ratio
        self.iters = iters
        self.value_int_coeff = value_int_coeff
        self.value_ext_coeff = value_ext_coeff
        self.ent_coeff = ent_coeff
        self.num_mini_batches = num_mini_batches

    def policy(self, obs):
        return self.model.policy(obs)

    def loss(
        self,
        returns_int,
        returns_ext,
        acts,
        advs,
        old_log_probs,
        obs,
        obs_mean,
        obs_std,
    ):
        current_obs = obs[:, :-1, ...]

        rews_int, X_r = self.model.intrinsic_rewards(obs, self.obs_mean, obs_std)
        policy, pred_values_int, pred_values_ext = self.model(current_obs)

        loss_int = 0.5 * ((pred_values_int - returns_int) ** 2).mean()
        loss_ext = 0.5 * ((pred_values_ext - returns_ext) ** 2).mean()

        # No coeff here, we try to learn value functions, not policy
        value_loss = loss_ext + loss_int

        pi_loss, kl, maxkl, clipfrac, entropy = ppo_loss(
            self.policy, current_obs, acts, advs, old_log_probs, self.clip_ratio
        )

        aux_loss = rews_int.mean()

        loss = pi_loss + value_loss + self.ent_coeff * entropy + aux_loss
        losses = {
            "value_ext": loss_ext,
            "value_int": loss_int,
            "vf": value_loss,
            "pi_loss": pi_loss,
            "ent": entropy,
            "clipfrac": clipfrac,
            "approx_kl": kl,
            "max_kl": maxkl,
            "aux_loss": aux_loss,
            "featvar": X_r.var(),
            "featmax": torch.abs(X_r).max(),
        }
        return loss, losses

    def update(self, episodes):
        # Remove last observation, it's not needed for the update
        obs = episodes.obs
        acts = episodes.acts

        B, N, *obs_shape = obs.shape

        batch_mean = obs.view(B * N, *obs_shape).mean(dim=0).clone().detach()
        batch_var = obs.view(B * N, *obs_shape).var(dim=0).clone().detach()
        batch_count = B * N
        with torch.no_grad():
            if not hasattr(self, "obs_mean"):
                self.obs_mean = batch_mean
                self.obs_var = batch_var
                self.obs_count = batch_count
            else:
                tot_count = batch_count + self.obs_count
                delta = batch_mean - self.obs_mean
                self.obs_mean += delta * batch_count / tot_count

                self.obs_count += batch_count
                m_a = self.obs_var * self.obs_count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + (delta) ** 2 * self.obs_count * batch_count / tot_count
                self.obs_var = M2 / tot_count

            obs_std = torch.sqrt(self.obs_var)

            # Model calls
            rews_int, X_r = self.model.intrinsic_rewards(obs, self.obs_mean, obs_std)
            old_dist, values_int, values_ext = self.model(obs)

            next_ext = values_ext[:, -1, ...]
            next_int = values_int[:, -1, ...]

            # Intrinsic reward is non-episodic, so all dones = 0
            dones = torch.zeros(*episodes.dones.shape)
            advantages = torch.zeros(*episodes.rews.shape)
            adv_int = gae_advantages(
                advantages, values_int, dones, rews_int, self.gamma, self.lambda_
            )
            adv_ext = episodes.gae_advantages(values_ext, self.gamma_ext, self.lambda_)

            advs = adv_int * self.value_int_coeff + adv_ext * self.value_ext_coeff

            old_dist = Categorical(logits=old_dist.logits[:, :-1, ...])
            old_log_probs = old_dist.log_prob(acts)

        returns_ext = episodes.discounted_returns(
            gamma=self.gamma_ext, pred_values=next_ext
        )

        dones = torch.zeros(*episodes.dones.shape)
        returns_int = torch.zeros(*episodes.rews.shape)
        returns_int = discounted_returns(
            returns_int,
            gamma=self.gamma,
            pred_values=next_int,
            dones=dones,
            rews=rews_int,
        )
        nperbatch = B // self.num_mini_batches

        info = dict(
            advmean=advs.mean(),
            advstd=advs.std(),
            retintmean=returns_int.mean(),  # previously retmean
            retintstd=returns_int.std(),  # previously retstd
            retextmean=returns_ext.mean(),  # previously not there
            retextstd=returns_ext.std(),  # previously not there
            rewintmean_unnorm=rews_int.mean(),  # previously rewmean
            rewintmax_unnorm=rews_int.max(),  # previously not there
            # rewintmean_norm=self.mean_int_rew,  # previously rewintmean
            # rewintmax_norm=self.max_int_rew,  # previously rewintmax
            # rewintstd_unnorm=rewstd,  # previously rewstd
            vpredintmean=values_int.mean(),  # previously vpredmean
            vpredintstd=values_int.std(),  # previously vrpedstd
            vpredextmean=values_ext.mean(),  # previously not there
            vpredextstd=values_ext.std(),  # previously not there
            ev_int=np.clip(
                explained_variance(
                    values_int[:, :-1, ...].contiguous().view(-1), returns_int.view(-1)
                ),
                -1,
                None,
            ),
            ev_ext=np.clip(
                explained_variance(
                    values_ext[:, :-1, ...].contiguous().view(-1), returns_ext.view(-1)
                ),
                -1,
                None,
            ),
        )

        info[f"mem_available"] = psutil.virtual_memory().available

        for i in range(self.iters):
            for start in range(0, B, nperbatch):
                end = start + nperbatch
                slice_ = slice(start, end)

                loss, losses = self.loss(
                    returns_int[slice_],
                    returns_ext[slice_],
                    acts[slice_],
                    advs[slice_],
                    old_log_probs[slice_],
                    obs[slice_],
                    self.obs_mean,
                    obs_std,
                )

                self.optimizer.zero_grad()
                loss.backward()
                total_norm = 0
                for p in self.model.parameters():
                    if p.requires_grad:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)
                losses["grad_norm"] = total_norm
                self.optimizer.step()

                if i == 0 and start == 0:
                    logger.debug("\t".join(f"{ln:>12}" for ln in losses))
                logger.debug("\t".join(f"{l:12.4f}" for l in losses.values()))
        return losses, info


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import solve, make_env

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", "--env", type=str, default="PitfallNoFrameskip-v4"
    )
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--clip-ratio", "--clip", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--value-ext-coeff", type=float, default=2)
    parser.add_argument("--value-int-coeff", type=float, default=1)
    parser.add_argument("--ent-coeff", type=float, default=1e-3)
    parser.add_argument("--target-kl", type=float, default=0.01)

    # 128 steps * 32 env in OpenAI
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num_mini_batches", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=int(1e9))
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gamma_ext", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger.info("Using PPO formulation of policy gradient.")

    env = make_env(args.env_name, args.num_envs, frame_stack=args.frame_stack)
    if isinstance(env.action_space, Discrete):
        n_acts = env.action_space.n
    elif isinstance(env.action_space, Box):
        assert (
            len(env.action_space.shape) == 1
        ), f"This example only works for envs with Box(n,) not {env.action_space} action spaces."
        n_acts = env.action_space.shape[0]

    global_model = GlobalModel(n_acts)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    assert (
        args.num_envs > args.num_mini_batches
    ), "We need more environments than minibatches."

    update = RDNValueUpdate(
        global_model,
        optimizer,
        args.gamma,
        args.gamma_ext,
        args.lam,
        args.clip_ratio,
        args.iters,
        args.value_int_coeff,
        args.value_ext_coeff,
        args.ent_coeff,
        args.num_mini_batches,
    )

    epochs = args.num_steps // args.batch_size + 1
    solve(args.env_name, env, update, logdir, epochs=epochs, batch_size=args.batch_size)

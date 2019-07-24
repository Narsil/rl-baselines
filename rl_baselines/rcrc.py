import multiprocessing
import torch.nn as nn
import torch
import cma
import tqdm
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from rl_baselines.core import logger, logdir
from gym.spaces import Discrete, Box
from torch.distributions import Normal


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 31, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 14, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 6, stride=2)

    def forward(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def init_W(n, m):
    weight = torch.normal(mean=torch.zeros((n, m)), std=torch.ones((n, m)))

    N = n * m
    p = int(0.2 * N)

    u, s, v = torch.svd(weight, compute_uv=True)
    s_ = 0.95 * s / s.max()

    weight = u * s_ * v.t()
    indices = np.random.choice(N, p)
    for i in indices:
        a = i // n
        b = i - a * n
        weight[a, b] = 0
    return weight


class FixedRandomModel(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.conv = Conv()
        self.W_in = nn.Linear(1152, 512, bias=False)
        self.W = nn.Linear(512, 512, bias=False)
        self.W.weight.data = init_W(512, 512)
        self.x_esn = None
        self.alpha = alpha

    def forward(self, obs):
        B = obs.shape[0]
        x_conv = self.conv(obs)
        x_conv_flat = x_conv.view(B, -1)

        if self.x_esn is None or self.x_esn.shape[0] != B:
            x_esn = torch.tanh(self.W_in(x_conv_flat))
        else:
            x_hat = torch.tanh(self.W_in(x_conv_flat) + self.W(self.x_esn))
            x_esn = (1 - self.alpha) * self.x_esn + self.alpha * x_hat
        self.x_esn = x_esn

        return (x_conv_flat, x_esn)


class RCRCUpdate(nn.Module):
    def __init__(self, model, num_envs, n_acts):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.es = cma.CMAEvolutionStrategy(
            np.zeros(1665 * n_acts), 0.1, {"popsize": num_envs}
        )

        self.W_outs = self.es.ask()
        self.W_out = nn.Parameter(torch.zeros(1665 * n_acts))
        self.n_acts = n_acts

    def policy(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        obs = obs / 255.0
        x_conv, x_esn = self.model(obs)
        B = obs.shape[0]

        S = torch.cat((x_conv, x_esn, torch.ones((B, 1))), dim=1)

        out = torch.zeros((B, self.n_acts))
        for i in range(B):
            if self.training:
                W_out = torch.from_numpy(
                    self.W_outs[i].reshape(1665, self.n_acts)
                ).float()
            else:
                W_out = self.W_out.reshape(1665, self.n_acts)
            A_hat = torch.matmul(S[i], W_out)
            A = torch.stack(
                (
                    torch.tanh(A_hat[0]),
                    (torch.tanh(A_hat[1]) + 1) / 2.0,
                    torch.clamp(torch.tanh(A_hat[2]), 0, 1),
                ),
                dim=0,
            )
            out[i, :] = A

        out = Normal(loc=out, scale=torch.ones(*out.shape) * 1e-8)
        return out

    def update(self, rewards):
        losses = {}
        info = {}
        self.es.tell(self.W_outs, -rewards)
        self.W_outs = self.es.ask()
        return losses, info


def run_full_episode(multi_env, policy):
    obs = env.reset()
    obs = torch.from_numpy(obs).float()

    dones = np.array([False] * multi_env.num_envs)
    rewards = np.array([0.0] * multi_env.num_envs)
    while not np.all(dones):
        dist = policy(obs)
        acts = dist.sample()
        obs, rews, ds, infos = env.step(acts.detach().numpy())
        obs = torch.from_numpy(obs).float()
        dones = dones + ds  # or
        rewards += rews * (1 - dones)

    return rewards


def solve(env_name, multi_env, policy_update, logdir, epochs, n_episodes):
    max_ret = -1e9
    writer = SummaryWriter(log_dir=logdir)

    # Weird bug, tensorboard sets its own root logger, we need to remove it.
    import logging

    root = logging.getLogger()
    root.handlers = []

    parameters = sum(p.numel() for p in policy_update.parameters() if p.requires_grad)
    logger.debug(f"Attempting to solve {env_name}")
    logger.debug(f"Epochs: {epochs}")
    logger.debug(f"Episodes per update: {n_episodes}")
    logger.debug(f"Policy Update: {policy_update}")
    logger.debug(f"Parameters: {parameters}")
    logger.debug(f"Reward threshold: {env.spec.reward_threshold}")
    for i in tqdm.tqdm(range(epochs)):

        rewards = np.zeros(multi_env.num_envs)
        for j in range(n_episodes):
            rews = run_full_episode(multi_env, policy_update.policy)
            rewards += rews
        rewards /= n_episodes

        aM = rewards.argmax(axis=0)
        M = rewards[aM]
        if M > max_ret:
            filename = os.path.join(logdir, "checkpoint.pth")
            policy_update.W_out = nn.Parameter(
                torch.from_numpy(policy_update.W_outs[aM]).float()
            )
            torch.save(policy_update, filename)

            logger.debug(f"Saved new best model: {filename}")
            max_ret = M
        writer.add_scalar(
            f"{env_name}/episode_reward_mean", rewards.mean(), global_step=i
        )
        writer.add_scalar(
            f"{env_name}/episode_reward_max", rewards.max(), global_step=i
        )
        writer.add_scalar(
            f"{env_name}/episode_reward_std", rewards.std(), global_step=i
        )

        if env.spec.reward_threshold and rewards.mean() > env.spec.reward_threshold:
            logger.info(f"{env_name}: Solved !")
            logger.info(
                f"{env_name}: Check out winning agent `python -m rl_baselines.test_agent --model={filename} --env={env_name}`"
            )
            return True

        policy_update.update(rewards)
        logger.debug(f"Rewards: {rewards}")
        logger.debug(f"Rewards mean: {rewards.mean()}")
    return False if env.spec.reward_threshold else None


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import make_env

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", "--env", type=str, default="CarRacing-v0")
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--n-episodes", type=int, default=4)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    logger.info("Using RCRC formulation.")

    env = make_env(args.env_name, args.num_envs)
    if isinstance(env.action_space, Discrete):
        n_acts = env.action_space.n
    elif isinstance(env.action_space, Box):
        assert (
            len(env.action_space.shape) == 1
        ), f"This example only works for envs with Box(n,) not {env.action_space} action spaces."
        n_acts = env.action_space.shape[0]

    fixed_model = FixedRandomModel(args.alpha)

    update = RCRCUpdate(fixed_model, args.num_envs, n_acts)
    update.train()

    solve(
        args.env_name,
        env,
        update,
        logdir,
        epochs=args.epochs,
        n_episodes=args.n_episodes,
    )

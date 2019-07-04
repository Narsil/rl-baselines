import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger("core")


class MLP(nn.Module):
    def __init__(self, sizes, activation=torch.tanh, out_activation=None):
        super().__init__()

        self.layers = nn.ModuleList()
        for in_, out_ in zip(sizes, sizes[1:]):
            layer = nn.Linear(in_, out_)
            self.layers.append(layer)
        self.activation = activation
        self.out_activation = out_activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
        if self.out_activation:
            x = self.out_activation(x)
        return x


class DiscretePolicy(nn.Module):
    def __init__(self, policy_model):
        super().__init__()
        self.model = policy_model

    def forward(self, state):
        logits = self.model(state)
        return Categorical(logits=logits)


class ContinuousPolicy(nn.Module):
    def __init__(self, policy_model, action_shape):
        super().__init__()
        self.model = policy_model

        # log_std = -0.5 -> std=0.6
        self.log_std = nn.Parameter(-0.5 * torch.ones(*action_shape))

    def forward(self, state):
        mu = self.model(state)
        return MultivariateNormal(mu, torch.diag(self.log_std.exp()))


class PolicyUpdate:
    def __init__(self, policy, optimizer, normalize_baseline):
        self.normalize_baseline = normalize_baseline
        self.policy = policy
        self.optimizer = optimizer

    def _baseline(self, episodes):
        """This is the policy baseline, but don't add normalization it is
        added in the `baseline` method."""
        raise NotImplementedError

    def baseline(self, episodes):
        batch_weights = self._baseline(episodes)
        weights = torch.Tensor(batch_weights)
        if self.normalize_baseline:
            weights = (weights - weights.mean()) / (weights.std() + 1e-5)
        return weights

    def loss(self, policy, episodes):
        raise NotImplementedError

    def update(self, episodes):
        self.optimizer.zero_grad()
        loss = self.loss(self.policy, episodes)
        loss.backward()
        self.optimizer.step()
        return loss

    def batch(self, episodes):
        batch_obs = [item for episode in episodes for item in episode.obs]
        batch_acts = [item for episode in episodes for item in episode.act]
        weights = self.baseline(episodes)
        obs = torch.Tensor(batch_obs)
        acts = torch.Tensor(batch_acts).unsqueeze(1)
        return obs, acts, weights


class Episode:
    def __init__(self):
        self.obs = []
        self.act = []
        self.rew = []

    def end(self):
        self.ret = sum(self.rew)
        self.len = len(self.rew)


def train_one_epoch(env, batch_size, render, policy, optimizer, policy_update):
    episodes = []
    episode = Episode()

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over

    # collect experience by acting in the environment with current policy
    step = 0
    while True:
        # save obs
        episode.obs.append(obs)

        # act in the environment
        dist = policy(torch.from_numpy(obs).float())
        act = dist.sample().item()
        obs, rew, done, _ = env.step(act)

        # save action, reward
        episode.act.append(act)
        episode.rew.append(rew)

        if done:
            # if episode is over, record info about episode
            episode.end()
            episodes.append(episode)

            # reset episode-specific variables
            obs, done = env.reset(), False
            episode = Episode()

            # end experience loop if we have enough of it
            if step > batch_size:
                break
        step += 1

    loss = policy_update.update(episodes)

    return loss, episodes


def create_models(env_name, hidden_sizes, lr):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, (Discrete, Box)
    ), "This example only works for envs with discrete/box action spaces."

    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        n_acts = env.action_space.n
        model = MLP(sizes=[obs_dim] + hidden_sizes + [n_acts])
        policy = DiscretePolicy(model)
    elif isinstance(env.action_space, Box):
        assert (
            len(env.action_space.shape) == 1
        ), "Can't handle Box action_shape with more than 1 dimension"
        n_acts = env.action_space.shape[0]
        model = MLP(sizes=[obs_dim] + hidden_sizes + [n_acts])
        policy = ContinuousPolicy(model, env.action_space.shape)
    else:
        raise NotImplementedError(
            "We don't handle action spaces different from box/discrete yet."
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    return env, policy, optimizer


def train(
    env_name,
    env,
    policy,
    optimizer,
    policy_update,
    epochs=100,
    batch_size=5000,
    render=False,
):

    writer = SummaryWriter()
    env_step = 0
    for epoch in range(epochs):
        batch_loss, episodes = train_one_epoch(
            env, batch_size, render, policy, optimizer, policy_update=policy_update
        )
        rets = np.mean([episode.ret for episode in episodes])
        lens = np.mean([episode.len for episode in episodes])
        logger.debug(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (epoch, batch_loss, rets, lens)
        )
        env_step += sum([episode.len for episode in episodes])
        writer.add_scalar(f"{env_name}/episode_reward", rets, global_step=env_step)
        writer.add_scalar(f"{env_name}/episode_length", lens, global_step=env_step)


def set_logger():
    import sys

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


if __name__ == "__main__":
    import argparse

    set_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    print("\nUsing simplest formulation of policy gradient.\n")

    hidden_sizes = [100]
    lr = 1e-2
    env, policy, optimizer = create_models(args.env_name, hidden_sizes, lr)
    policy_update = VPGUpdate(policy, optimizer, normalize_baseline=True)
    train(env, policy, optimizer, policy_update)

import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


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


class VPGUpdate(PolicyUpdate):
    def _baseline(self, episodes):
        weights = []
        for episode in episodes:
            ret = 0
            returns = []
            for rew in reversed(episode.rew):
                ret += rew
                returns.append(ret)
            weights += list(reversed(returns))
        return weights

    def loss(self, policy, episodes):
        batch_obs = [item for episode in episodes for item in episode.obs]
        batch_acts = [item for episode in episodes for item in episode.act]

        weights = self.baseline(episodes)

        dist = policy(torch.Tensor(batch_obs))
        log_probs = dist.log_prob(torch.Tensor(batch_acts))
        loss = -((weights * log_probs).mean())
        return loss


def train(
    env_name="CartPole-v0",
    hidden_sizes=[100],
    lr=1e-2,
    epochs=100,
    batch_size=5000,
    render=False,
):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    model = MLP(sizes=[obs_dim] + hidden_sizes + [n_acts])
    policy = DiscretePolicy(model)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    policy_update = VPGUpdate(policy, optimizer, normalize_baseline=True)
    # training loop
    for i in range(epochs):
        batch_loss, episodes = train_one_epoch(
            env, batch_size, render, policy, optimizer, policy_update=policy_update
        )
        batch_rets = [episode.ret for episode in episodes]
        batch_lens = [episode.len for episode in episodes]
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    print("\nUsing simplest formulation of policy gradient.\n")
    train(env_name=args.env_name, render=args.render, lr=args.lr)

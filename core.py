import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import logging
import os


def set_logger(logger):
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)
    root.handlers = []

    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    import sys

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    import os
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join("runs", current_time)

    os.makedirs(logdir, exist_ok=True)
    filename = os.path.join(logdir, "run.log")
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logdir


logger = logging.getLogger("rl-baselines")
if not logger.handlers:
    logdir = set_logger(logger)


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


class Baseline:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _get(self, episodes):
        raise NotImplementedError

    def __call__(self, episodes):
        batch_weights = self._get(episodes)
        weights = torch.Tensor(batch_weights)
        if self.normalize:
            weights = (weights - weights.mean()) / (weights.std() + 1e-5)
        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize={self.normalize})"


class FullReturnBaseline(Baseline):
    def _get(self, episodes):
        weights = [episode.ret for episode in episodes for _ in episode.rew]
        return weights


class FutureReturnBaseline(Baseline):
    def _get(self, episodes):
        weights = []
        for episode in episodes:
            ret = 0
            returns = []
            for rew in reversed(episode.rew):
                ret += rew
                returns.append(ret)
            weights += list(reversed(returns))
        return weights


def discounted_cumsum(elements, gamma):
    discounted = []
    cur = 0
    for element in reversed(elements):
        cur = element + gamma * cur
        discounted.append(cur)
    return list(reversed(discounted))


class DiscountedReturnBaseline(Baseline):
    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def _get(self, episodes):
        weights = []
        for episode in episodes:
            weights += discounted_cumsum(episode.rew, self.gamma)
        return weights


class GAEBaseline(Baseline):
    def __init__(self, value_model, gamma, lambda_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.value_model = value_model
        assert gamma > lambda_, f"For stability λ({lambda_}) < γ({gamma}) is advised"

    def _get(self, episodes):
        obs = [item for episode in episodes for item in episode.obs]
        with torch.no_grad():
            values = self.value_model(torch.Tensor(obs))
        start = 0
        weights = []
        for episode in episodes:
            end = start + len(episode.obs)
            # End value always 0 as we never cut episodes.
            v_pi = torch.cat((values[start:end], torch.Tensor([[0]])), dim=0)
            deltas = (
                torch.Tensor(episode.rew).unsqueeze(1)
                + self.gamma * v_pi[1:]
                - v_pi[:-1]
            )
            assert list(deltas.shape) == [len(episode.obs), 1], "Dimension sanity check"
            weights += discounted_cumsum(deltas, self.gamma * self.lambda_)
            start = end
        return torch.Tensor(weights)


class ModelUpdate(nn.Module):
    loss_name = "loss"

    def __init__(self, model, optimizer, baseline, iters=1):
        super().__init__()
        self.baseline = baseline
        self.model = model
        self.optimizer = optimizer
        self.iters = iters

    def loss(self, model, episodes):
        raise NotImplementedError

    def update(self, episodes):
        for i in range(self.iters):
            self.optimizer.zero_grad()
            loss = self.loss(self.model, episodes)
            loss.backward()
            self.optimizer.step()
        return {self.loss_name: loss}

    def batch(self, episodes):
        batch_obs = [item for episode in episodes for item in episode.obs]
        batch_acts = [item for episode in episodes for item in episode.act]
        weights = self.baseline(episodes)
        obs = torch.Tensor(batch_obs)
        acts = torch.stack(batch_acts, dim=0)
        return obs, acts, weights


class PolicyUpdate(ModelUpdate):
    loss_name = "policy_loss"

    @property
    def policy(self):
        return self.model


class ValueUpdate(ModelUpdate):
    loss_name = "value_loss"

    @property
    def value(self):
        return self.model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.baseline, DiscountedReturnBaseline
        ), "Value models need to learn discounted returns"

    def loss(self, model, episodes):
        obs, _, returns = self.batch(episodes)
        values = model(obs)
        return F.mse_loss(values, returns.unsqueeze(1))


class ActorCriticUpdate(nn.Module):
    def __init__(self, policy_update, value_update):
        super().__init__()
        assert isinstance(policy_update, PolicyUpdate)
        assert isinstance(value_update, ValueUpdate)
        self.policy_update = policy_update
        self.value_update = value_update

    @property
    def policy(self):
        return self.policy_update.policy

    @property
    def value(self):
        return self.value_update.value

    def update(self, episodes):
        losses = {}
        p_losses = self.policy_update.update(episodes)
        v_losses = self.value_update.update(episodes)
        losses.update(p_losses)
        losses.update(v_losses)
        return losses


class Episode:
    def __init__(self):
        self.obs = []
        self.act = []
        self.rew = []

    def end(self):
        self.ret = sum(self.rew)
        self.len = len(self.rew)
        assert len(self.rew) == len(self.obs) == len(self.act)

    def __len__(self):
        return self.len


def train_one_epoch(env, batch_size, render, policy_update):
    episodes = []
    episode = Episode()

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over

    policy = policy_update.policy
    # collect experience by acting in the environment with current policy
    step = 0
    while True:
        # save obs
        episode.obs.append(obs)

        # act in the environment
        dist = policy(torch.from_numpy(obs).float())
        act = dist.sample()
        obs, rew, done, _ = env.step(act.numpy())

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

    losses = policy_update.update(episodes)

    return losses, episodes


def create_models(env_name, hidden_sizes, pi_lr, vf_lr):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, (Discrete, Box)
    ), "This example only works for envs with discrete/box action spaces."

    assert (
        len(env.observation_space.shape) == 1
    ), f"This example only works for envs with Box(n,) not {env.observation_space} observation spaces."
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
    poptimizer = torch.optim.Adam(policy.parameters(), lr=pi_lr)
    value = MLP(sizes=[obs_dim] + hidden_sizes + [1])
    voptimizer = torch.optim.Adam(value.parameters(), lr=vf_lr)
    return env, (policy, poptimizer), (value, voptimizer)


def solve(
    env_name, env, policy_update, logdir, epochs=100, batch_size=5000, render=False
):

    writer = SummaryWriter(log_dir=logdir)
    env_step = 0

    # Weird bug, tensorboard sets its own root logger, we need to remove it.
    root = logging.getLogger()
    root.handlers = []

    logger.debug(f"Attempting to solve {env_name}")
    logger.debug(f"Epochs: {epochs}")
    logger.debug(f"Batch_size: {batch_size}")
    logger.debug(f"Policy Update: {policy_update}")
    logger.debug(f"Reward threshold: {env.spec.reward_threshold}")

    max_ret = -1e9

    for epoch in range(epochs):
        losses, episodes = train_one_epoch(
            env, batch_size, render, policy_update=policy_update
        )
        rets = np.mean([episode.ret for episode in episodes])
        lens = np.mean([episode.len for episode in episodes])
        loss_string = "\t".join(
            [f"{loss_name}: {loss:.3f}" for loss_name, loss in losses.items()]
        )
        logger.debug(
            "epoch: %3d \t %s \t return: %.3f \t ep_len: %.3f"
            % (epoch, loss_string, rets, lens)
        )
        env_step += sum([episode.len for episode in episodes])
        writer.add_scalar(f"{env_name}/episode_reward", rets, global_step=env_step)
        writer.add_scalar(f"{env_name}/episode_length", lens, global_step=env_step)
        if env.spec.reward_threshold and rets > env.spec.reward_threshold:
            logger.info(f"{env_name}: Solved !")
            return True
        if rets > max_ret:
            filename = os.path.join(logdir, "checkpoint.pth")
            torch.save(policy_update, filename)
            logger.debug(f"Saved new best model: {filename}")
            max_ret = rets
    return False if env.spec.reward_threshold else None

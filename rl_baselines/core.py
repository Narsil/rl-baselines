import numpy as np
from gym.spaces import Discrete, Box
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import datetime
import os
import sys
from collections import deque
from rl_baselines.environment import SubprocVecEnv, make_single_env
from rl_baselines.models import ContinuousPolicy, DiscretePolicy, MLP, Conv, ValueModel


def set_logger(logger):
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)
    root.handlers = []

    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
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


def gae_advantages(advantages, values, dones, rews, gamma, lambda_):
    lastgaelam = 0
    num_steps = advantages.shape[1]

    assert num_steps == dones.shape[1]
    assert num_steps + 1 == values.shape[1]
    assert num_steps <= rews.shape[1]

    for t in range(num_steps - 1, -1, -1):  # nsteps-1 ... 0
        nextdone = dones[:, t + 1] if t + 1 < num_steps else 0
        nextvals = values[:, t + 1]
        nextnotdone = 1 - nextdone
        try:
            delta = rews[:, t] + gamma * nextvals * nextnotdone - values[:, t]
            advantages[:, t] = lastgaelam = (
                delta + gamma * lambda_ * nextnotdone * lastgaelam
            )
        except Exception:
            import ipdb

            ipdb.set_trace()
    return advantages


def discounted_returns(rets, gamma, pred_values, dones, rews):
    num_steps = rets.shape[1]
    assert num_steps == dones.shape[1]
    assert num_steps <= rews.shape[1]

    curr_rets = pred_values
    for t in range(num_steps - 1, -1, -1):  # nsteps-1 ... 0
        nextdone = dones[:, t + 1] if t + 1 < num_steps else 0
        nextnotdone = 1 - nextdone
        curr_rets = rews[:, t] + gamma * curr_rets * nextnotdone
        rets[:, t] = curr_rets
    return rets


class Episodes:
    def __init__(self, num_env, num_steps, obs_shape, act_shape):
        self.num_steps = num_steps
        self.num_env = num_env
        self.obs = torch.zeros((num_env, num_steps + 1, *obs_shape))
        self.acts = torch.zeros((num_env, num_steps, *act_shape))
        self.rews = torch.zeros((num_env, num_steps))
        self.dones = torch.zeros((num_env, num_steps))

    def get_buffer(self, name):
        if not hasattr(self, name):
            setattr(self, name, torch.zeros((self.num_env, self.num_steps)))
        return getattr(self, name)

    def gae_advantages(self, values, gamma, lambda_):
        advantages = self.get_buffer("advs")
        return gae_advantages(advantages, values, self.dones, self.rews, gamma, lambda_)

    def discounted_returns(self, gamma, pred_values):
        rets = self.get_buffer("rets")
        return discounted_returns(rets, gamma, pred_values, self.dones, self.rews)

    def discounted_returns_discard(self, gamma):
        rets = self.get_buffer("rets")
        curr_rets = self.rews[:, self.num_steps - 1]
        # Starts are tracking if we are looking at a complete episode.
        # Basically we want to discard
        masks = torch.zeros(*rets.shape).long()
        for t in range(self.num_steps - 1, -1, -1):  # nsteps-1 ... 0
            nextdone = (
                self.dones[:, t + 1]
                if t + 1 < self.num_steps
                else torch.zeros(*curr_rets.shape)
            )
            nextnotdone = 1 - nextdone
            masks[t] = (
                (masks[t + 1] | nextdone.long())
                if t + 1 < self.num_steps
                else nextdone.long()
            )
            curr_rets = self.rews[:, t] + gamma * curr_rets * nextnotdone
            rets[:, t] = curr_rets
        return rets, masks

    def stats(self):
        stats = torch.zeros((self.num_env, 3))
        # 0: actual return, 1: length, 2: is it a full_episode
        all_returns = []
        all_lengths = []
        for t in range(self.num_steps - 2, -1, -1):
            for i in range(self.num_env):
                if self.dones[i, t]:
                    if stats[i, 2]:
                        all_returns.append(stats[i, 0])
                        all_lengths.append(stats[i, 1])
                    stats[i, 2] = 1
            nextdone = self.dones[:, t + 1] if t + 1 < self.num_steps else 0
            nextnotdone = 1 - nextdone
            stats[:, 0] = self.rews[:, t] + nextnotdone * stats[:, 0]
            stats[:, 1] = 1 + nextnotdone * stats[:, 1]
        # for i in range(self.num_env):
        #     if stats[i, 2]:
        #         all_returns.append(stats[i, 0])
        #         all_lengths.append(stats[i, 1])
        return np.array(all_returns), np.array(all_lengths)


def gather_episodes(episodes, env, num_steps, policy, epoch):

    # reset episode-specific variables

    # obs = env.reset()  # first obs comes from starting distribution
    if epoch == 0:
        obs = env.reset()  # first obs comes from starting distribution
    else:
        obs, _, _, _ = env.step_wait()

    obs = torch.from_numpy(obs).float()

    finished_episodes_stats = []

    for step in range(num_steps):
        episodes.obs[:, step] = obs
        # act in the environment
        dist = policy(obs)
        acts = dist.sample()
        obs, rews, dones, infos = env.step(acts.cpu().numpy())
        obs = torch.from_numpy(obs).float()

        episodes.acts[:, step] = acts  # Already torch tensor
        episodes.rews[:, step] = torch.from_numpy(rews)
        episodes.dones[:, step] = torch.from_numpy(dones)
        for info in infos:
            if "episode" in info:
                finished_episodes_stats.append(info["episode"])

    # Push last observation needed for advantages
    episodes.obs[:, num_steps] = obs

    # Send a last action to environments so we can continue the episode
    dist = policy(obs)
    acts = dist.sample()
    env.step_async(acts.cpu().numpy())

    return episodes, finished_episodes_stats


def train_one_epoch(env, num_steps, epoch, episodes, policy_update, device):

    policy = policy_update.policy

    # collect experience by acting in the environment with current policy
    # start = datetime.datetime.now()
    episodes, finished_episodes_stats = gather_episodes(
        episodes, env, num_steps, policy, epoch
    )
    # end_rollout = datetime.datetime.now()

    losses, infos = policy_update.update(episodes)

    # logger.debug(f"Rollout time {end_rollout - start}")
    # logger.debug(f"Update time {datetime.datetime.now() - end_rollout}")

    return losses, infos, episodes, finished_episodes_stats


def default_model(env, hidden_sizes, n_acts):
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, (Discrete, Box)
    ), "This example only works for envs with discrete/box action spaces."

    assert len(env.observation_space.shape) in [
        1,
        3,
    ], f"This example only works for envs with Box(n,) or Box(h, w, c) not {env.observation_space} observation spaces."
    if len(env.observation_space.shape) == 1:
        obs_dim = env.observation_space.shape[0]
        model = MLP(sizes=[obs_dim] + hidden_sizes + [n_acts])
    elif len(env.observation_space.shape) == 3:
        model = Conv(
            input_shape=env.observation_space.shape, sizes=hidden_sizes + [n_acts]
        )
    return model


def make_env(env_name, num_envs, **kwargs):
    env = SubprocVecEnv(
        [lambda: make_single_env(env_name, **kwargs) for i in range(num_envs)]
    )
    return env


def default_policy_model(env, hidden_sizes):
    # make environment, check spaces, get obs / act dims

    if isinstance(env.action_space, Discrete):
        n_acts = env.action_space.n
    elif isinstance(env.action_space, Box):
        assert (
            len(env.action_space.shape) == 1
        ), f"This example only works for envs with Box(n,) not {env.action_space} action spaces."
        n_acts = env.action_space.shape[0]
    model = default_model(env, hidden_sizes, n_acts)
    if isinstance(env.action_space, Discrete):
        policy = DiscretePolicy(model)
    elif isinstance(env.action_space, Box):
        policy = ContinuousPolicy(model, env.action_space.shape)
    else:
        raise NotImplementedError(
            "We don't handle action spaces different from box/discrete yet."
        )
    return policy


def create_models(env, hidden_sizes, pi_lr, vf_lr):
    policy = default_policy_model(env, hidden_sizes)

    poptimizer = torch.optim.Adam(policy.parameters(), lr=pi_lr)

    value = ValueModel(default_model(env, hidden_sizes, 1))
    voptimizer = torch.optim.Adam(value.parameters(), lr=vf_lr)
    return (policy, poptimizer), (value, voptimizer)


def solve(
    env_name,
    env,
    policy_update,
    logdir,
    epochs=100,
    batch_size=5000,
    render=False,
    device=None,
):
    if device is None:
        device = "cpu"

    writer = SummaryWriter(log_dir=logdir)
    env_step = 0

    # Weird bug, tensorboard sets its own root logger, we need to remove it.
    root = logging.getLogger()
    root.handlers = []

    parameters = sum(p.numel() for p in policy_update.parameters())
    logger.debug(f"Attempting to solve {env_name}")
    logger.debug(f"Epochs: {epochs}")
    logger.debug(f"Batch_size: {batch_size}")
    logger.debug(f"Policy Update: {policy_update}")
    logger.debug(f"Parameters: {parameters}")
    logger.debug(f"Reward threshold: {env.spec.reward_threshold}")

    max_ret = 0

    num_steps = batch_size // env.num_envs
    episodes = Episodes(
        env.num_envs, num_steps, env.observation_space.shape, env.action_space.shape
    )

    episode_stats = deque(maxlen=100)
    for epoch in range(epochs):
        losses, infos, episodes, finished_episodes_stats = train_one_epoch(
            env, num_steps, epoch, episodes, policy_update=policy_update, device=device
        )

        episode_stats.extend(finished_episodes_stats)
        all_rewards = []
        all_lengths = []
        visited_rooms = set()
        for stat in episode_stats:
            all_rewards.append(stat["reward"])
            all_lengths.append(stat["length"])
            if "visited_rooms" in stat:
                visited_rooms |= stat["visited_rooms"]

        rets = np.array(all_rewards).mean()
        lens = np.array(all_lengths).mean()

        loss_string = "\t".join(
            [f"{loss_name}: {loss:.3f}" for loss_name, loss in losses.items()]
        )
        logger.debug(
            "epoch: %3d \t %s \t return: %.3f \t ep_len: %.3f"
            % (epoch, loss_string, rets, lens)
        )
        if visited_rooms:
            logger.debug(f"Visited rooms : {visited_rooms}")
        env_step += num_steps * env.num_envs

        for loss_name, loss in losses.items():
            writer.add_scalar(
                f"{env_name}-losses/{loss_name}", loss, global_step=env_step
            )

        for info_name, info in infos.items():
            writer.add_scalar(
                f"{env_name}-info/{info_name}", info, global_step=env_step
            )
        writer.add_scalar(f"{env_name}-info/episode_reward", rets, global_step=env_step)
        writer.add_scalar(
            f"{env_name}-info/episode_best_reward", max_ret, global_step=env_step
        )
        writer.add_scalar(f"{env_name}-info/episode_length", lens, global_step=env_step)
        if visited_rooms:
            writer.add_scalar(
                f"{env_name}-info/visited_rooms",
                len(visited_rooms),
                global_step=env_step,
            )

        if rets > max_ret:
            filename = os.path.join(logdir, "checkpoint.pth")
            torch.save(policy_update, filename)
            logger.debug(f"Saved new best model: {filename}")
            max_ret = rets

        if env.spec.reward_threshold and rets > env.spec.reward_threshold:
            logger.info(f"{env_name}: Solved !")
            logger.info(
                f"{env_name}: Check out winning agent `python -m rl_baselines.test_agent --model={filename} --env={env_name}`"
            )
            return True
    return False if env.spec.reward_threshold else None

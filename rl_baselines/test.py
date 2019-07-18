import unittest
import logging
import multiprocessing
import torch
from rl_baselines.core import create_models, logdir, solve, logger, Episodes, make_env
from rl_baselines.baselines import (
    FutureReturnBaseline,
    DiscountedReturnBaseline,
    GAEBaseline,
)
from rl_baselines.model_updates import ActorCriticUpdate, ValueUpdate
from rl_baselines.reinforce import REINFORCE
from rl_baselines.ppo import PPO


logger.setLevel(logging.CRITICAL)


def discounted_cumsum(elements, gamma):
    discounted = []
    cur = 0
    for element in reversed(elements):
        cur = element + gamma * cur
        discounted.append(cur)
    return list(reversed(discounted))


class TestBaselines(unittest.TestCase):
    def setUp(self):
        episodes = Episodes(num_env=2, num_steps=4, obs_shape=(2,), act_shape=())
        # There is one last observation for advantage estimation.
        episodes.obs[0] = torch.Tensor([[1, -2], [1, -2], [1, -2], [1, -2], [1, -2]])
        episodes.rews[0] = torch.Tensor([1, 0, 1, 0])
        episodes.dones[0] = torch.Tensor([0, 0, 0, 0])
        episodes.acts[0] = torch.Tensor([1, 1, 1, 1])

        episodes.obs[1] = torch.Tensor(
            [[10, -20], [10, -20], [10, -20], [-10, 20], [-10, 20]]
        )
        episodes.rews[1] = torch.Tensor([1, 1, 1, 1])
        episodes.acts[1] = torch.Tensor([1, 1, 1, 1])
        episodes.dones[1] = torch.Tensor([0, 1, 0, 0])

        self.episodes = episodes

    def test_gae_return_baseline(self):
        gamma = 0.99
        lambda_ = 0.95
        values = torch.arange(1, 11).reshape((2, 5)).float()
        weights = self.episodes.gae_advantages(values, gamma, lambda_)

        rewards = self.episodes.rews
        deltas = torch.Tensor(
            [
                [
                    rewards[0, 0] + (values[0, 1] * gamma - values[0, 0]),
                    rewards[0, 1] + (values[0, 2] * gamma - values[0, 1]),
                    rewards[0, 2] + (values[0, 3] * gamma - values[0, 2]),
                    rewards[0, 3] + (values[0, 4] * gamma - values[0, 3]),
                ],
                [
                    # XXX: cut episode here
                    rewards[1, 0] + (-values[1, 0]),
                    rewards[1, 1] + (values[1, 2] * gamma - values[1, 1]),
                    rewards[1, 2] + (values[1, 3] * gamma - values[1, 2]),
                    rewards[1, 3] + (values[1, 4] * gamma - values[1, 3]),
                ],
            ]
        )
        target = torch.Tensor(
            [
                discounted_cumsum(deltas[0], gamma * lambda_),  # First episode
                discounted_cumsum(deltas[1, :1], gamma * lambda_)
                + discounted_cumsum(deltas[1, 1:], gamma * lambda_),  # Second episode
            ]
        )
        self.assertEqual(weights.tolist(), target.tolist())


class TestVanilla(unittest.TestCase):
    num_envs = 1

    def get_update_model(self, env):
        hidden_sizes = [100]
        lr = 1e-2
        gamma = 0.99
        lam = 0.97

        (policy, optimizer), (value, vopt) = create_models(env, hidden_sizes, lr, lr)
        baseline = GAEBaseline(value, gamma=gamma, lambda_=lam)
        policy_update = REINFORCE(policy, optimizer, baseline)

        vbaseline = DiscountedReturnBaseline(gamma=gamma, normalize=False)
        value_update = ValueUpdate(value, vopt, vbaseline, iters=1)
        return ActorCriticUpdate(policy_update, value_update)

    def test_cartpole_v0(self):
        env_name = "CartPole-v0"
        env = make_env(env_name, 1)
        policy_update = self.get_update_model(env)
        result = solve(env_name, env, policy_update, logdir)
        self.assertEqual(result, True)

    def test_cartpole_v1(self):
        env_name = "CartPole-v1"
        env = make_env(env_name, 1)
        policy_update = self.get_update_model(env)
        result = solve(env_name, env, policy_update, logdir)
        self.assertEqual(result, True)

    def test_lunar_lander_v2(self):
        env_name = "LunarLander-v2"
        env = make_env(env_name, 1)
        policy_update = self.get_update_model(env)
        result = solve(env_name, env, policy_update, logdir, epochs=500)
        self.assertEqual(result, True)


if __name__ == "__main__":
    unittest.main()

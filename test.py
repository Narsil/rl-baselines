import unittest
import logging
import torch
from core import (
    create_models,
    logdir,
    solve,
    logger,
    FullReturnBaseline,
    FutureReturnBaseline,
    DiscountedReturnBaseline,
    GAEBaseline,
    discounted_cumsum,
    Episode,
)
from vanilla import VPGUpdate
from ppo import PPO


logger.setLevel(logging.CRITICAL)


class TestUtils(unittest.TestCase):
    def test_discount_cumsum(self):
        elements = [1, 1]
        self.assertEqual(discounted_cumsum(elements, 0.9), [1.9, 1])

        elements = [1, 2, 3]
        self.assertEqual(discounted_cumsum(elements, 0.9), [5.23, 4.7, 3])

        elements = []
        self.assertEqual(discounted_cumsum(elements, 0.9), [])


class TestBaselines(unittest.TestCase):
    def setUp(self):
        ep1 = Episode()
        ep2 = Episode()

        ep1.obs.append([1, -2])
        ep1.obs.append([1, -2])
        ep1.obs.append([-1, -2])

        ep1.rew.append(1)
        ep1.rew.append(0)
        ep1.rew.append(0)

        ep1.act.append(1)
        ep1.act.append(1)
        ep1.act.append(1)

        ep2.obs.append([10, -20])
        ep2.obs.append([-10, -20])
        ep2.obs.append([-5, 10])

        ep2.rew.append(1)
        ep2.rew.append(1)
        ep2.rew.append(0)

        ep2.act.append(1)
        ep2.act.append(1)
        ep2.act.append(0)

        ep1.end()
        ep2.end()

        self.episodes = [ep1, ep2]

    def test_full_return_baseline(self):
        baseline = FullReturnBaseline(normalize=False)
        weights = baseline(self.episodes)

        target = torch.Tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.assertEqual(weights.tolist(), target.tolist())

        baseline = FullReturnBaseline(normalize=True)
        weights = baseline(self.episodes)

        target2 = (target - target.mean()) / (target.std() + 1e-5)
        self.assertEqual(weights.tolist(), target2.tolist())

    def test_future_return_baseline(self):
        baseline = FutureReturnBaseline(normalize=False)
        weights = baseline(self.episodes)

        target = torch.Tensor([1.0, 0.0, 0.0, 2.0, 1.0, 0.0])
        self.assertEqual(weights.tolist(), target.tolist())

        baseline = FutureReturnBaseline(normalize=True)
        weights = baseline(self.episodes)

        target2 = (target - target.mean()) / (target.std() + 1e-5)
        self.assertEqual(weights.tolist(), target2.tolist())

    def test_discounted_return_baseline(self):
        gamma = 0.99
        baseline = DiscountedReturnBaseline(gamma=gamma, normalize=False)
        weights = baseline(self.episodes)

        target = torch.Tensor(
            [
                1.0 + 0.0 * gamma,
                0.0 + 0.0 * gamma,
                0.0,
                1.0 + 1.0 * gamma,
                1.0 + 0.0 * gamma,
                0.0,
            ]
        )
        self.assertEqual(weights.tolist(), target.tolist())

        baseline = DiscountedReturnBaseline(gamma=gamma, normalize=True)
        weights = baseline(self.episodes)

        target2 = (target - target.mean()) / (target.std() + 1e-5)
        self.assertEqual(weights.tolist(), target2.tolist())

    def test_gae_return_baseline(self):
        gamma = 0.99
        lambda_ = 0.95
        values = torch.arange(1, 7).reshape((6, 1)).float()
        value_model = lambda x: values
        baseline = GAEBaseline(
            value_model=value_model, gamma=gamma, lambda_=lambda_, normalize=False
        )
        weights = baseline(self.episodes)

        rewards = [item for episode in self.episodes for item in episode.rew]
        deltas = torch.Tensor(
            [
                rewards[0] + (values[1] * gamma - values[0]),
                rewards[1] + (values[2] * gamma - values[1]),
                rewards[2] + (0.0 * gamma - values[2]),
                rewards[3] + (values[4] * gamma - values[3]),
                rewards[4] + (values[5] * gamma - values[4]),
                rewards[5] + (0.0 * gamma - values[5]),
            ]
        )
        target = torch.Tensor(
            discounted_cumsum(deltas[:3], gamma * lambda_)  # First episode
            + discounted_cumsum(deltas[3:], gamma * lambda_)  # Second episode
        )
        self.assertEqual(weights.tolist(), target.tolist())

        baseline = GAEBaseline(
            value_model=value_model, gamma=gamma, lambda_=lambda_, normalize=True
        )
        weights = baseline(self.episodes)

        target2 = (target - target.mean()) / (target.std() + 1e-5)
        self.assertEqual(weights.tolist(), target2.tolist())


class TestVanilla(unittest.TestCase):
    hidden_sizes = [100]
    lr = 1e-2

    def test_cartpole_v0(self):
        env_name = "CartPole-v0"
        env, policy, optimizer = create_models(env_name, self.hidden_sizes, self.lr)
        baseline = FutureReturnBaseline()
        policy_update = VPGUpdate(policy, optimizer, baseline)
        result = solve(env_name, env, policy_update, logdir)
        self.assertEqual(result, True)

    def test_cartpole_v1(self):
        env_name = "CartPole-v1"
        env, policy, optimizer = create_models(env_name, self.hidden_sizes, self.lr)
        baseline = FutureReturnBaseline()
        policy_update = VPGUpdate(policy, optimizer, baseline)
        result = solve(env_name, env, policy_update, logdir)
        self.assertEqual(result, True)

    def test_inverted_pendulum_v2(self):
        env_name = "InvertedPendulum-v2"
        env, policy, optimizer = create_models(env_name, self.hidden_sizes, self.lr)
        baseline = FutureReturnBaseline()
        policy_update = VPGUpdate(policy, optimizer, baseline)
        result = solve(env_name, env, policy_update, logdir)
        self.assertEqual(result, True)

    def test_lunar_lander_v2(self):
        env_name = "LunarLander-v2"
        env, policy, optimizer = create_models(env_name, self.hidden_sizes, self.lr)
        baseline = FullReturnBaseline()
        policy_update = VPGUpdate(policy, optimizer, baseline)
        result = solve(env_name, env, policy_update, logdir, epochs=500)
        self.assertEqual(result, True)


if __name__ == "__main__":
    unittest.main()

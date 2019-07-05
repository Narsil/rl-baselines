import unittest
import logging
from core import (
    create_models,
    logdir,
    FutureReturnBaseline,
    solve,
    logger,
    FullReturnBaseline,
)
from vanilla import VPGUpdate
from ppo import PPO


logger.setLevel(logging.CRITICAL)


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
        result = solve(env_name, env, policy_update, logdir)
        self.assertEqual(result, True)


if __name__ == "__main__":
    unittest.main()

import argparse
import torch
import gym

from core import logger

# Necessary for lazy torch.load
from vanilla import VPGUpdate
from ppo import PPO


def test_agent(env_name, policy_update_filename):
    policy_update = torch.load(policy_update_filename)
    logger.debug(f"Loaded : {policy_update}")
    env = gym.make(env_name)
    obs = env.reset()
    done = False
    policy = policy_update.policy
    total_reward = 0
    steps = 0
    while not done:
        env.render()
        dist = policy(torch.from_numpy(obs).float())
        act = dist.sample()
        obs, rew, done, _ = env.step(act.numpy())
        total_reward += rew
        steps += 1

    logger.debug(f"Total reward : {total_reward}")
    logger.debug(f"Episode length : {steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-update", "--model", type=str)
    parser.add_argument("--env-name", "--env", type=str)

    args = parser.parse_args()

    test_agent(args.env_name, args.policy_update)

from vanilla import train
import logging
import gym

if __name__ == "__main__":
    import argparse

    logger = logging.getLogger("ALL")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    env_names = gym.envs.registry.env_specs.keys()
    for env_name in env_names:
        try:
            train(env_name=env_name, lr=args.lr)
        except Exception as e:
            logger.error(f"{env_name}: {e}")

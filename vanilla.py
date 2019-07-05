from core import PolicyUpdate, logger, logdir


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
        obs, acts, weights = self.batch(episodes)

        dist = policy(obs)
        log_probs = dist.log_prob(acts)
        loss = -((weights * log_probs).mean())
        return loss


if __name__ == "__main__":
    import argparse
    from core import solve, create_models, FullReturnBaseline, FutureReturnBaseline

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    logger.info("Using vanilla formulation of policy gradient.")

    hidden_sizes = [100]
    lr = 1e-2
    env, policy, optimizer = create_models(args.env_name, hidden_sizes, lr)
    baseline = FutureReturnBaseline()
    policy_update = VPGUpdate(policy, optimizer, baseline)
    solve(args.env_name, env, policy_update, logdir, epochs=args.epochs)

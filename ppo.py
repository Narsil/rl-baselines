from core import PolicyUpdate
import torch
import logging

logger = logging.getLogger("ppo")


class PPO(PolicyUpdate):
    def __init__(self, policy_iters, clip_ratio, target_kl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_iters = policy_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

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

    def loss(self, policy, episodes, obs, acts, weights, old_log_probs):
        clip_ratio = self.clip_ratio
        dist = policy(obs)
        log_probs = dist.log_prob(acts)

        ratio = (log_probs - old_log_probs).exp()
        approx_kl = (old_log_probs - log_probs).mean().item()
        clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        loss = -(torch.min(ratio * weights, clipped * weights)).mean()
        return loss, approx_kl

    def update(self, episodes):
        obs, acts, weights = self.batch(episodes)
        with torch.no_grad():
            old_dist = self.policy(obs)
            old_log_probs = old_dist.log_prob(acts)
        for i in range(self.policy_iters):
            self.optimizer.zero_grad()
            loss, kl = self.loss(
                self.policy, episodes, obs, acts, weights, old_log_probs
            )
            if kl > self.target_kl:
                logger.warning(
                    f"Stopping after {i} iters because KL > {self.target_kl}"
                )
                break
            loss.backward()
            self.optimizer.step()
        return loss


if __name__ == "__main__":
    import argparse
    from core import set_logger, train, create_models

    set_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--clip-ratio", "--clip", type=float, default=0.2)
    parser.add_argument("--policy-iters", type=int, default=80)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    logger.info("Using PPO formulation of policy gradient.")

    hidden_sizes = [100]
    lr = 1e-2
    env, policy, optimizer = create_models(args.env_name, hidden_sizes, lr)
    policy_update = PPO(
        args.policy_iters,
        args.clip_ratio,
        args.target_kl,
        policy,
        optimizer,
        normalize_baseline=True,
    )
    train(args.env_name, env, policy, optimizer, policy_update)

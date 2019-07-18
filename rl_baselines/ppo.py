from rl_baselines.core import logger, logdir
from rl_baselines.model_updates import PolicyUpdate
import multiprocessing
import torch


def ppo_loss(policy, obs, acts, weights, old_log_probs, clip_ratio):
    dist = policy(obs)
    log_probs = dist.log_prob(acts)

    diff = log_probs - old_log_probs
    ratio = (diff).exp()
    approx_kl = (diff ** 2).mean().item()
    max_kl = (diff ** 2).max().item()
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    clipfrac = ((ratio > (1 + clip_ratio)) | (ratio < (1 - clip_ratio))).float().mean()

    loss = -(torch.min(ratio * weights, clipped * weights)).mean()
    entropy = dist.entropy().mean()
    return loss, approx_kl, max_kl, clipfrac, entropy


class PPO(PolicyUpdate):
    def __init__(self, policy_iters, clip_ratio, target_kl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_iters = policy_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

    def update(self, episodes):
        obs, acts, weights = self.batch(episodes)

        with torch.no_grad():
            old_dist = self.policy(obs)
            old_log_probs = old_dist.log_prob(acts)
        # logger.debug("\t".join(["pi_loss", "kl", "clipfrac", "entropy"]))
        for i in range(self.policy_iters):
            self.optimizer.zero_grad()
            loss, kl, max_kl, clipfrac, entropy = ppo_loss(
                self.policy, obs, acts, weights, old_log_probs, self.clip_ratio
            )
            if kl > self.target_kl:
                logger.warning(
                    f"Stopping after {i} iters because KL > {self.target_kl}"
                )
                break
            # logger.debug("\t".join("%.4f" % v for v in [loss, kl, clipfrac, entropy]))
            loss.backward()
            self.optimizer.step()
        infos = {}
        return {"ppo_loss": loss}, infos


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import solve, create_models, make_env
    from rl_baselines.baselines import GAEBaseline, DiscountedReturnBaseline
    from rl_baselines.model_updates import ActorCriticUpdate, ValueUpdate

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--clip-ratio", "--clip", type=float, default=0.2)
    parser.add_argument("--policy-iters", type=int, default=80)
    parser.add_argument("--value-iters", type=int, default=80)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--pi-lr", type=float, default=3e-4)
    parser.add_argument("--vf-lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger.info("Using PPO formulation of policy gradient.")

    hidden_sizes = [100]
    env = make_env(args.env_name, args.num_envs)
    (policy, optimizer), (value, vopt) = create_models(
        env, hidden_sizes, args.pi_lr, args.vf_lr
    )

    baseline = GAEBaseline(value, gamma=args.gamma, lambda_=args.lam)
    policy_update = PPO(
        args.policy_iters, args.clip_ratio, args.target_kl, policy, optimizer, baseline
    )

    vbaseline = DiscountedReturnBaseline(gamma=args.gamma, normalize=False)
    value_update = ValueUpdate(value, vopt, vbaseline, iters=args.value_iters)
    update = ActorCriticUpdate(policy_update, value_update)
    solve(
        args.env_name,
        env,
        update,
        logdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

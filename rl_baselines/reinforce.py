from rl_baselines.core import logger, logdir
from rl_baselines.model_updates import PolicyUpdate
import multiprocessing


class REINFORCE(PolicyUpdate):
    def update(self, episodes):
        obs, acts, weights = self.batch(episodes)

        dist = self.model(obs)
        log_probs = dist.log_prob(acts)
        loss = -((weights * log_probs).mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        infos = {}
        return {"reinforce_loss": loss}, infos


if __name__ == "__main__":
    import argparse
    from rl_baselines.core import solve, create_models, make_env
    from rl_baselines.baselines import GAEBaseline, DiscountedReturnBaseline
    from rl_baselines.model_updates import ActorCriticUpdate, ValueUpdate

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--num-envs", type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    logger.info("Using vanilla formulation of policy gradient.")

    hidden_sizes = [100]
    lr = 1e-2
    env = make_env(args.env_name, args.num_envs)
    (policy, optimizer), (value, vopt) = create_models(
        env, hidden_sizes, args.lr, args.lr
    )
    baseline = GAEBaseline(value, gamma=args.gamma, lambda_=args.lam)
    policy_update = REINFORCE(policy, optimizer, baseline)

    vbaseline = DiscountedReturnBaseline(gamma=args.gamma, normalize=False)
    value_update = ValueUpdate(value, vopt, vbaseline, iters=1)
    update = ActorCriticUpdate(policy_update, value_update)
    solve(
        args.env_name,
        env,
        policy_update,
        logdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

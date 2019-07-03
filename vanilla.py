import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation), x


class MLP(nn.Module):
    def __init__(self, sizes, activation=torch.tanh, out_activation=None):
        super().__init__()

        self.layers = nn.ModuleList()
        for in_, out_ in zip(sizes, sizes[1:]):
            layer = nn.Linear(in_, out_)
            with torch.no_grad():
                layer.bias.zero_()

            self.layers.append(layer)
        self.activation = activation
        self.out_activation = out_activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
        if self.out_activation:
            x = self.out_activation(x)
        return x


def train(
    env_name="CartPole-v0",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=100,
    batch_size=5000,
    render=False,
):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits, prev_layer = mlp(obs_ph, sizes=hidden_sizes + [n_acts])
    model = MLP(sizes=[obs_dim] + hidden_sizes + [n_acts])
    model

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    model.layers[0].weight = nn.Parameter(torch.from_numpy(values[0].T))
    model.layers[0].bias = nn.Parameter(torch.from_numpy(values[1]))
    model.layers[1].weight = nn.Parameter(torch.from_numpy(values[2].T))
    model.layers[1].bias = nn.Parameter(torch.from_numpy(values[3]))

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch(
            env,
            sess,
            obs_ph,
            act_ph,
            weights_ph,
            actions,
            batch_size,
            loss,
            train_op,
            render,
            model,
            optimizer,
            action_masks,
            logits,
            log_probs,
            prev_layer,
        )
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )


# for training policy
def train_one_epoch(
    env,
    sess,
    obs_ph,
    act_ph,
    weights_ph,
    actions,
    batch_size,
    loss,
    train_op,
    render,
    model,
    optimizer,
    action_masks_op,
    logits_op,
    log_probs_op,
    prev_layer_op,
):
    # make some empty lists for logging.
    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:

        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        # logits = model(torch.from_numpy(obs).float())
        # dist = Categorical(logits=logits)
        # act = dist.sample().item()
        # obs, rew, done, _ = env.step(act)
        act = sess.run(actions, {obs_ph: obs.reshape(1, -1)})[0]
        obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    OBS = np.array(batch_obs).astype(np.float32)
    WEIGHTS = np.array(batch_weights).astype(np.float32)
    ACTS = np.array(batch_acts).astype(np.int)
    optimizer.zero_grad()
    B = len(batch_acts)
    action_masks = torch.zeros((B, 2))
    action_masks.scatter_(1, torch.from_numpy(ACTS).unsqueeze(1), 1)
    logits = model(torch.from_numpy(OBS))
    log_probs = (action_masks * F.log_softmax(logits, dim=-1)).sum(dim=1)
    weights = torch.from_numpy(WEIGHTS)
    batch_loss = -((weights * log_probs).mean())

    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)

    tf_logits = sess.run([logits_op], feed_dict={obs_ph: OBS})[0]
    tf_loss = sess.run(
        [loss], feed_dict={obs_ph: OBS, act_ph: ACTS, weights_ph: WEIGHTS}
    )
    X = sess.run(
        [prev_layer_op], feed_dict={obs_ph: OBS, act_ph: ACTS, weights_ph: WEIGHTS}
    )[0]
    Y = sess.run(
        [logits_op], feed_dict={obs_ph: OBS, act_ph: ACTS, weights_ph: WEIGHTS}
    )[0]
    XX = model.activation(model.layers[0](torch.from_numpy(OBS))).detach().numpy()
    YY = model.layers[1](torch.from_numpy(XX)).detach().numpy()
    print(f"tf loss: {tf_loss}, torch loss: {batch_loss}")
    print(f"X: {(X - XX).mean()}")
    print(f"Y: {(Y - YY).mean()}")
    print(f"A: {(model.layers[0].weight.detach().numpy() - values[0].T).mean()}")
    print(f"B: {(model.layers[0].bias.detach().numpy() - values[1]).mean()}")
    print(f"C: {(model.layers[1].weight.detach().numpy() - values[2].T).mean()}")
    print(f"D: {(model.layers[1].bias.detach().numpy() - values[3]).mean()}")
    print(f"logits: {(logits.detach().numpy() - tf_logits).mean()}")
    tf_loss, _ = sess.run(
        [loss, train_op],
        feed_dict={
            obs_ph: np.array(batch_obs),
            act_ph: np.array(batch_acts),
            weights_ph: np.array(batch_weights),
        },
    )

    batch_loss.backward()
    optimizer.step()

    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    print(f"tf loss: {tf_loss}, torch loss: {batch_loss}")
    print(f"A: {(model.layers[0].weight.detach().numpy() - values[0].T).mean()}")
    print(f"B: {(model.layers[0].bias.detach().numpy() - values[1]).mean()}")
    print(f"C: {(model.layers[1].weight.detach().numpy() - values[2].T).mean()}")
    print(f"D: {(model.layers[1].bias.detach().numpy() - values[3]).mean()}")
    return batch_loss, batch_rets, batch_lens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    print("\nUsing simplest formulation of policy gradient.\n")
    train(env_name=args.env_name, render=args.render, lr=args.lr)

import torch.nn as nn
import torch.nn.functional as F
import torch
from rl_baselines.baselines import DiscountedReturnBaseline


class ModelUpdate(nn.Module):
    loss_name = "loss"

    def __init__(self, model, optimizer, baseline, iters=1):
        super().__init__()
        self.baseline = baseline
        self.model = model
        self.optimizer = optimizer
        self.iters = iters

    def update(self, episodes):
        raise NotImplementedError

    def batch(self, episodes):
        weights = self.baseline(episodes)
        # Remove last observation, it's not needed for the update anymore
        return episodes.obs[:, :-1, ...], episodes.acts, weights


class PolicyUpdate(ModelUpdate):
    loss_name = "policy_loss"

    @property
    def policy(self):
        return self.model


class ValueUpdate(ModelUpdate):
    loss_name = "value_loss"

    @property
    def value(self):
        return self.model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.baseline, DiscountedReturnBaseline
        ), "Value models need to learn discounted returns"
        self.gamma = self.baseline.gamma

    def update(self, episodes):
        with torch.no_grad():
            pred_values = self.model(episodes.obs[:, -1, ...])
        weights = episodes.discounted_returns(gamma=self.gamma, pred_values=pred_values)

        # Remove last observation, it's not needed for the update anymore
        obs, returns = episodes.obs[:, :-1, ...], weights

        for i in range(self.iters):
            values = self.model(obs)
            loss = F.mse_loss(values, returns)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {self.loss_name: loss}


class ActorCriticUpdate(nn.Module):
    def __init__(self, policy_update, value_update):
        super().__init__()
        assert isinstance(policy_update, PolicyUpdate)
        assert isinstance(value_update, ValueUpdate)
        self.policy_update = policy_update
        self.value_update = value_update

    @property
    def policy(self):
        return self.policy_update.policy

    @property
    def value(self):
        return self.value_update.value

    def update(self, episodes):
        losses = {}
        p_losses, infos = self.policy_update.update(episodes)
        v_losses = self.value_update.update(episodes)
        losses.update(p_losses)
        losses.update(v_losses)
        return losses, infos

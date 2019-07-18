import torch
import numpy as np


class Baseline:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _get(self, episodes):
        raise NotImplementedError

    def __call__(self, episodes):
        batch_weights = self._get(episodes)
        weights = torch.Tensor(batch_weights)
        if self.normalize:
            weights = (weights - weights.mean()) / (weights.std() + 1e-5)
        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize={self.normalize})"


class DiscountedReturnBaseline(Baseline):
    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def _get(self, episodes):
        weights, masks = episodes.discounted_returns_discard(self.gamma)
        return weights


class FutureReturnBaseline(DiscountedReturnBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(gamma=1, *args, **kwargs)


class GAEBaseline(Baseline):
    def __init__(self, value_model, gamma, lambda_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.value_model = value_model
        assert gamma > lambda_, f"For stability λ({lambda_}) < γ({gamma}) is advised"

    def _get(self, episodes):
        with torch.no_grad():
            values = self.value_model(episodes.obs)

        weights = episodes.gae_advantages(values, self.gamma, self.lambda_)
        return weights

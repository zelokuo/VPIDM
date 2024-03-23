import abc

import torch
import numpy as np

from sgmse.util.registry import Registry


PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")


@PredictorRegistry.register('euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        f, g = self.rsde.sde(x, t, *args)
        x_mean = x + f * dt
        x = x_mean + g[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@PredictorRegistry.register('reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False,):
        super().__init__(sde, score_fn, probability_flow=probability_flow)


    def update_fn(self, x, t, *args):
        f, g = self.rsde.discretize(x, t, *args)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + g[:, None, None, None] * z
        return x, x_mean


@PredictorRegistry.register('reverse_diffusion_predict_mean')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)
        self.score_fn = score_fn


    def update_fn(self, x, t, y, tn, mean, smooth=0.05):
        f, _ = self.sde.discretize(x, t, y)
        g = self.sde._std(tn) 
        z = torch.randn_like(x)
        alpha = self.sde._alpha(tn)
        x_p = - self.score_fn(x, t, y)
        mean = smooth*mean + (1 - smooth)*alpha*x_p
        x_mean = mean - f
        x = x_mean + g * z
        return x, x_mean, x_p



@PredictorRegistry.register('reverse_controlable_denoising')
class ReverseControlableDenoising(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)
        self.score_fn = score_fn


    def update_fn(self, x, t, y, s=0.5, rcm=False, delta=0, *args):
        if rcm :
            t = t - delta
            x_p = -self.score_fn(x, t, y)
            x = x
        else:
            x_p = -self.score_fn(y, t, y)
            alpha = self.sde._alpha(t)
            x = s*x + (1 - s)*alpha*x_p
            f, g = self.sde.discretize(x, t, y, *args)
            x = x - f
        return x, x_p




@PredictorRegistry.register('none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t, *args):
        return x, x

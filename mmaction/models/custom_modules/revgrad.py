import torch
import torch.nn as nn
from torch.autograd import Function


class RevGrad(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def update_alpha(self, alpha):
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGradFunc.apply(input_, self._alpha)
    
    @property
    def alpha(self):
        return self._alpha


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None

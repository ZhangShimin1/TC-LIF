import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

tab4_str = '\t\t\t\t'  # used for aligning code
curly_bracket_l = '{'
curly_bracket_r = '}'


@torch.jit.script
def heaviside(x: torch.Tensor):
    '''
    * :ref:`API in English <heaviside.__init__-en>`
    .. _heaviside.__init__-cn:

    :param x: 输入tensor
    :return: 输出tensor

    heaviside阶跃函数，定义为

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    阅读 `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_ 以获得更多信息。

    * :ref:`中文API <heaviside.__init__-cn>`
    .. _heaviside.__init__-en:

    :param x: the input tensor
    :return: the output tensor

    The heaviside function, which is defined by

    .. math::
        g(x) =
        \\begin{cases}
        1, & x \\geq 0 \\\\
        0, & x < 0 \\\\
        \\end{cases}

    For more information, see `HeavisideStepFunction <https://mathworld.wolfram.com/HeavisideStepFunction.html>`_.

    '''
    return (x >= 0).to(x)


@torch.jit.script
def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    x_abs = x.abs()
    mask = (x_abs > (1 / alpha))
    grad_x = (grad_output * (- (alpha ** 2) * x_abs + alpha)).masked_fill_(mask, 0)
    return grad_x, None


class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def piecewise_exp_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 * (- alpha * x.abs()).exp_() * grad_output, None


class piecewise_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_exp_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def soft_sign_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (2 * alpha * (1 / alpha + x.abs()).pow_(2)), None


class soft_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return soft_sign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def nonzero_sign_log_abs_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (1 / alpha + x.abs()), None


class nonzero_sign_log_abs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return nonzero_sign_log_abs_backward((grad_output, ctx.saved_tensors[0], ctx.alpha))


@torch.jit.script
def erf_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output * (- (x * alpha).pow_(2)).exp_() * (alpha / math.sqrt(math.pi)), None


class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return erf_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def piecewise_leaky_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, w: float, c: float):
    mask_width = (x.abs() < w)
    mask_c = mask_width.logical_not()
    return grad_output * x.masked_fill(mask_width, 1 / w).masked_fill(mask_c, c), None, None


class piecewise_leaky_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w=1, c=0.01):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.w = w
            ctx.c = c
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_leaky_relu_backward(grad_output, ctx.saved_tensors[0], ctx.w, ctx.c)


class squarewave_fourier_series(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, n: int, T_period: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.n = n
            ctx.T_period = T_period
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = 0.
        x = ctx.saved_tensors[0]
        w = math.pi * 2. / ctx.T_period
        for i in range(1, ctx.n):
            grad_x += torch.cos_((2 * i - 1.) * w * x)

        grad_x *= 4. / ctx.T_period
        grad_x *= grad_output

        return grad_x, None, None


class s2nn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, beta: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.beta = beta
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sgax = torch.sigmoid(ctx.alpha * x)
        grad_x = torch.where(x < 0., ctx.alpha * sgax * (1. - sgax), ctx.beta / (x + 1.))
        return grad_x * grad_output, None, None


class q_pseudo_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        x = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_x = ((1 + 2 / (ctx.alpha - 1) * x.abs()).pow_(-ctx.alpha)) * grad_output
        return grad_x, None


@torch.jit.script
def leaky_k_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, leak: float, k: float):
    mask1 = (x >= 0.).to(x)
    grad_x = mask1 * k + (1. - mask1) * leak
    return grad_output * grad_x, None, None


class leaky_k_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, leak, k):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.leak = leak
            ctx.k = k
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return leaky_k_relu_backward(grad_output, ctx.saved_tensors[0], ctx.leak, ctx.k)


@torch.jit.script
def fake_numerical_gradient_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    grad_x = torch.clamp_max(((x >= 0.) * 2. - 1.) / x, alpha)
    return grad_output * grad_x, None


class fake_numerical_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return fake_numerical_gradient_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


@torch.jit.script
def log_tailed_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    mask_gt1 = x > 1.
    mask_le0 = x <= 0.
    grad_x = torch.ones_like(grad_output)
    grad_x[mask_gt1] = 1. / x[mask_gt1]
    grad_x[mask_le0] = alpha
    return grad_output * grad_x, None


class log_tailed_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return log_tailed_relu_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class SingleExponential(torch.autograd.Function):
    """Surrogate gradients for standard binary spikes"""

    @staticmethod
    def forward(
            ctx,
            input
    ):
        ctx.save_for_backward(input.clone())
        return input.ge(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input * torch.exp(-torch.abs(input - threshold)), grad_input * -torch.exp(-torch.abs(input - threshold)), None
        return grad_input * torch.exp(-torch.abs(input))


class Triangle(torch.autograd.Function):
    """Altered from code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
    max(0, 1 − |ui[t] − θ|)"""

    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class Rectangle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        spikes = input.ge(0.).float()
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5

        return grad_input * temp.float()

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

gamma = 0.5  # gradient scale
lens = 0.5
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        #temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma
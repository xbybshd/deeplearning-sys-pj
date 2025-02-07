"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        # grad 是一个中转临时变量，并且构造的时候不加梯度，结束就销毁，即使创建计算图，也是在内部。而外部param全部是用 .data，所以就避免了扩增
        for param in self.params:
            if param not in self.u.keys():
                self.u[param] = ndl.zeros_like(param.grad, requires_grad = False)
            grad = self.u[param].data * self.momentum + (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)
            grad = ndl.Tensor(grad, dtype = param.dtype, requires_grad=False)
            self.u[param] = grad
            param.data = param.data - self.lr * grad.data



    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                grad = param.grad.data + self.weight_decay * param.data
                self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad.data
                self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * grad.data * grad.data
                u_hat = self.m[param] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param] / (1 - self.beta2 ** self.t)
                out = param.data - self.lr * u_hat / (ndl.ops.power_scalar(v_hat, 0.5) + self.eps)
                param.data = ndl.Tensor(out, dtype=param.dtype).data

"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 流程
        # X(batch x w_in) * W(w_in x w_out) + b(batch x w_out) 其中b的batch维度是 1 x w_out广播而来

        self.weight = Parameter(init.kaiming_uniform(in_features,out_features,requires_grad = True), device=device);
        self.has_bias = bias
        if self.has_bias:
            # 这里bias接的是w的输出，它本身的输出维度是1，（w_out,1)维向量  计算时要将(w_out,1)变成(1,w_out)广播成(batch,w_out)
            self.bias = Parameter(init.kaiming_uniform(out_features,1,requires_grad = True).transpose(), device=device);

    def forward(self, X: Tensor) -> Tensor:
        out = X.matmul(self.weight)
        if self.has_bias:
            boradcast_shape = [1] * (len(out.shape) - 1) + [self.out_features]
            bias = self.bias.reshape(boradcast_shape).broadcast_to(out.shape)
            out += bias
        return out


class Flatten(Module):
    def forward(self, X):
        batch_size = X.shape[0]
        dim = 1
        for i in range(1,len(X.shape)):
            dim *= X.shape[i]
        return ops.reshape(X, (batch_size, dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for module in self.modules:
            input = module(input)
        return input


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # 只是2维的情况， log（sumexp） - y(true label)
        softmax = ops.logsumexp(logits,axes = (1,))
        batch_size, class_num = logits.shape
        y_one_hot = init.one_hot(class_num, y, logits.device)
        I = ops.summation(logits*y_one_hot, axes=(1,))
        return ops.summation(softmax - I) / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, dim, device = device, dtype = dtype),device = device, dtype = dtype)
        self.bias = Parameter(init.zeros(1, dim, device = device, dtype = dtype),device = device, dtype = dtype)
        self.running_mean = init.zeros(dim, device = device, dtype = dtype)
        self.running_var = init.ones(dim, device = device, dtype = dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_size, feature_size = x.shape
            mean = (x.sum(axes = (0,))/batch_size).reshape((1,feature_size))
            var = (((x-mean.broadcast_to(x.shape))**2).sum(axes=(0,))/batch_size).reshape((1,feature_size))
            self.running_mean = self.running_mean*(1 - self.momentum) + mean.reshape(self.running_mean.shape)*self.momentum
            self.running_var = self.running_var*(1 - self.momentum) + var.reshape(self.running_var.shape)*self.momentum
            mean = mean.broadcast_to(x.shape)
            var = var.broadcast_to(x.shape)
            std_x = (x - mean) / (ops.power_scalar((var + self.eps), 0.5))
            weight = self.weight.broadcast_to(x.shape)
            bias = self.bias.broadcast_to(x.shape)
            return std_x * weight + bias
        else:
            std_x = (x - self.running_mean.broadcast_to(x.shape)) / ops.power_scalar(self.running_var.broadcast_to(x.shape) + self.eps, 0.5)
            return std_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1,dim,device = device, dtype = dtype),device = device, dtype = dtype)
        self.bias = Parameter(init.zeros(1,dim,device = device, dtype = dtype),device = device, dtype = dtype)        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        old_shape = x.shape[:]
        input_shape = x.shape
        if(len(input_shape)> 2):
            batch_size = 1
            for i in range(0, len(input_shape) - 1):
                batch_size *= input_shape[i]
            x = x.reshape((batch_size, input_shape[-1]))
        batch_size, feature_size = x.shape
        mean = ((x.sum(axes = (1,))) / feature_size).reshape((batch_size,1)).broadcast_to(x.shape)
        var = (((x-mean)**2).sum(axes = (1,))/feature_size).reshape((batch_size,1)).broadcast_to(x.shape)
        std_x = (x - mean) / (ops.power_scalar((var + self.eps), 0.5))
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        res = std_x * weight + bias
        if len(old_shape) > 2:
            res = res.reshape(old_shape)
        return res
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        else:
            mask = init.randb(*x.shape, p = 1 -self.p, dtype=x.dtype, device=x.device)
            return (x * mask) / (1 -self.p)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)

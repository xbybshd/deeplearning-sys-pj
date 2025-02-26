from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        if self.axes is None:
            self.axes = tuple(list(range(len(Z.shape))))
        max_z_k = array_api.max(Z, axis = self.axes, keepdims = True) # 维持维度不变，所有元素取最大
        max_z_nk = array_api.max(Z, axis = self.axes, keepdims = False) # 维持维度不变，所有元素取最大
        # max_z_t = array_api.max(Z, axis=self.axes, keepdims=True) # 公式最后的那个maxz
        max_z_t = array_api.broadcast_to(max_z_k, Z.shape)
        return array_api.log(array_api.sum(array_api.exp(Z-max_z_t), axis = self.axes)) + max_z_nk # 最后计算求和维度会收为一维 batch x 1

    def gradient(self, out_grad, node):  # 此处公式可见推导 https://www.zhouxin.space/notes/gradient-of-log-sum-exp/
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        z = node.inputs[0] # batch x m x n
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]  # batch x m x 1  假设轴是2
        grad = exp(z - node.reshape(shape).broadcast_to(z.shape)) # node batch -> batch x m x 1 -> batch x m x n  直接广播是对不齐的
        # 或者出错，或者把batch维度当成最内层维度从右往左广播
        return out_grad.reshape(shape).broadcast_to(z.shape) * grad # out_grad batch -> batch x m x 1 -> batch x m x n 



def logsumexp(a, axes=None):
    print("logsumexp.axes:",axes)
    return LogSumExp(axes=axes)(a)

"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b

        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # if not isinstance(node.inputs[0], NDArray) or not isinstance(
        #     node.inputs[1], NDArray
        # ):
        #     raise ValueError("Both inputs must be tensors (NDArray).")
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a **(b-1))
        grad_b = out_grad * (a ** b) * (log(a.data))
        return grad_a, grad_b
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return self.scalar * (power_scalar(a, self.scalar-1)) * out_grad

        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b 

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b = node.inputs
        return out_grad/b, -a*out_grad/(b*b)

        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar

        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            ax0, ax1 = self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        permute_axes = list(range(a.ndim))
        permute_axes[ax0], permute_axes[ax1] = ax1, ax0
        return a.permute(permute_axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)

        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)

        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 获取输入张量的原始形状
        ori_shape = node.inputs[0].shape

        # 创建一个列表，用于标记哪些维度需要在反向传播中“缩减”
        shrink_dims = [i for i in range(len(self.shape))]
        # 遍历输入和广播后的形状，判断哪些维度需要进行求和（即“缩减”）
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1  # 如果对应的维度大小相同，就不需要缩减

        # 过滤掉所有不需要缩减的维度（即保留需要求和的维度）
        shrink_dims = list(filter(lambda x: x >= 0, shrink_dims))
        
        # 将输出的梯度按照需要缩减的维度进行求和，并且重新调整形状为原始输入张量的形状,如果是多个维度广播,需要调用多次求和
        for i in shrink_dims:
            out_grad = out_grad.sum(i)
            for j in range(len(shrink_dims)):
                if shrink_dims[j] > i:
                    shrink_dims[j] -= 1
        return out_grad.reshape(ori_shape)

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 获取算子输入矩阵的维度形状，是一个list，元素是每维度
        new_shape = list(node.inputs[0].shape)
        # 类似三元运算符，如果此算子求和时没标明维度，说明会对所有维度都求和，如果标明维度，那就只求和标明的维度
        axes = range(len(new_shape)) if self.axes is None else self.axes
        # 对要求和的维度里，在new_shape list中标明1
        if isinstance(axes,int):
            axes = [axes]
        for axis in axes:
            new_shape[axis] = 1
        # 被求和的维度在输出向量的形状中是被删除了，new_shape是在反向回来的梯度矩阵上进行reshape，把删除的维度恢复为1，之后再1的基础上
        # 广播出真正的维度
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)

        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # return array_api.matmul(a, b)
        return a@b

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        # 先保证最后两个维度相同，前面的成为批量维度，批量维度不同，则对小的数组加一维度进行拓展
        # 如果维度相同，且一个数组维度数值是1，则进行拓展，除此之外认为错误
        # 广播机制：
        # 如果两个数组的维度不同，NumPy 会在维度较小的数组的前面加上 1，直到两个数组的维度相同。
        # 在对齐维度时，如果某个维度的大小是 1，NumPy 会通过扩展该维度来匹配另一个数组的维度。
        # 如果两个数组在某一维度上的大小不相同，并且该维度上的大小都不为 1，则广播会失败，抛出错误。
        # 当因为维度数量不同而拓展时，在反向要将这一维度reduce掉
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad

        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return - a

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return - out_grad

        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]

        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * array_api.exp(node.inputs[0].realize_cached_data())

        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        zeros = array_api.full(a.shape, 0, dtype=a.dtype, device = a.device)
        return array_api.maximum(a, zeros)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # realize_cached_data 是因为我要numpy,copy是因为需要直接在数据上修改，用out是因为如果用input，要再处理一遍小于0
        # out 所有大于0的元素，为ture，也构成了索引，然后索引里面为1的都是1
        relu_mask = Tensor(node.inputs[0].cached_data > 0, device=node.inputs[0].device)
        return out_grad * relu_mask

        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_data = node.inputs[0].realize_cached_data()
        # return out_grad * (-(input_data ** 2) + 1)
        return out_grad * (1 - (array_api.tanh(input_data) ** 2))
        # return out_grad * add_scalar(-(node ** 2), 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(args) > 0:
            shape = args[0].shape
            for arg in args:
                assert arg.shape == shape, "The shape of all tensors should be the same"
            ret_shape = list(shape)
            ret_shape.insert(self.axis, len(args))
            ret = array_api.empty(ret_shape, device=args[0].device)
            for i, arg in enumerate(args):
                slices = [slice(None)] * len(ret_shape)
                slices[self.axis] = i
                ret[tuple(slices)] = arg
            return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        ret = []
        ret_shape = list(A.shape)
        ret_shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            ret.append((A[tuple(slices)]).compact().reshape(ret_shape))
        return tuple(ret)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        if isinstance(axes, list):
            axes = tuple(axes)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        out_shape = list(a.shape)
        for i in self.axes:
            out_shape[i] *= self.dilation + 1
        out = array_api.full(out_shape, 0, device=a.device)
        slices = [slice(None)] * len(a.shape)
        for dim in self.axes:
            slices[dim] = slice(None, None, self.dilation+1)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        out_shape = list(a.shape)
        for i in self.axes:
            out_shape[i] //= self.dilation + 1
        out = array_api.empty(out_shape, device=a.device)
        slices = [slice(None)] * len(a.shape)
        for dim in self.axes:
            slices[dim] = slice(None, None, self.dilation+1)
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().permute(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        index = [0] * len(self.axes)
        for i in range(len(self.axes)):
            index[self.axes[i]] = i
        return permute(out_grad, tuple(index))
        ### END YOUR SOLUTION
        
def permute(a, axes):
    return Permute(axes)(a)

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        assert len(A.shape) == 4, "The input tensor should be 4D"
        assert len(B.shape) == 4, "The kernel tensor should be 4D"
        A = A.compact()
        B = B.compact()
        batch_size, in_height, in_width, in_channel = A.shape
        bs, hs, ws, cs = A.strides
        kernel_height, kernel_width, in_channel, out_channel = B.shape
        
        
        
        pad_A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))).compact()
        batch_size, in_height, in_width, in_channel = pad_A.shape
        bs, hs, ws, cs = pad_A.strides
        receiptive_field_shape = (batch_size, (in_height - kernel_height) // self.stride + 1, (in_width - kernel_width) // self.stride + 1, kernel_height, kernel_width, in_channel)
        receiptive_field_strides = (bs, hs * self.stride, ws * self.stride, hs, ws, cs)
        receiptive_field = pad_A.as_strided(receiptive_field_shape, receiptive_field_strides).compact()
        reveiptive_vector = receiptive_field.reshape((receiptive_field.size //(kernel_height * kernel_width * in_channel), kernel_height * kernel_width * in_channel)).compact()
        kernel_vector = B.reshape((kernel_height * kernel_width * in_channel, out_channel)).compact()
        out = reveiptive_vector @ kernel_vector
        out = out.reshape((batch_size, (in_height - kernel_height) // self.stride + 1, (in_width - kernel_width) // self.stride + 1, out_channel)).compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        s, _, _, _ = W.shape
        
        # 计算X_grad
        W_flipped = flip(W, (0, 1))
        W_flipped_permuted = transpose(W_flipped, (2, 3)) # transpose 只支持两个维度的交换
        outgrad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        X_grad = conv(outgrad_dilated, W_flipped_permuted, padding=s - 1 - self.padding)
        
        # 计算W_grad
        # outgrad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        outgrad_dilated_permuted = permute(outgrad_dilated, (1, 2, 0, 3))
        X_permuted = permute(X, (3, 1, 2, 0))
        W_grad = conv(X_permuted, outgrad_dilated_permuted, padding=self.padding)
        W_grad = permute(W_grad, (1, 2, 0, 3))
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



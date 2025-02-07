## q1  NDarray backend  
1.首先要把hw2,hw3中实现的一些函数复制过来,在ops_mathematic.py中,hw2实现的函数有些是给予numpy的,但是在hw4中要替换为ndarray.py,具体就是将array.api的函数,替换为运算符或者是ndarry.py中本来就有的,且ndarray头文件要加上
```python
from .backend_selection import Device, cpu, all_devices
```  
以保证我们这里使用的后端是根据环境变量切换  
2. logsumexp在测试文件中,会有输入是(1,1,1)但是axis为空的情况,在max函数中并没有处理,而是在test中对对照组torch做了处理,所以要把对称轴为空的情况,在logsumexp中不全  函数中,对于重新构造shape的代码 
```python  
shape = tuple(((s.stop - s.start + s.step - 1) // s.step) for s in idxs)
```  
之前加了一个max,然后有些值就变成了0,删去之后可以正常通过,不知道为什么  
3. max 函数在测试文件中会存在输入int的情况,需要直接加上特判
4. gradient的函数中,涉及input计算的要用cache_data,运算符与前向函数一样用array_api的
这一行如果直接看flat是有数值的,但是加上下表就是空值,导致增减eps完全无效

## q2  CIFAR-10 dataset  
将hw2中的几个文件复制到hw4中,因为测试文件的问题,默认cpubackend是np.cpu,并不是ndl.cpu,所以需要修改Dataloader的__getitem__函数,显示传参ndl.cpu()(这种可能实现不太好,但是可以过测试,因为np.cpu没有许多功能)  
```python
    def __next__(self):
        ### BEGIN YOUR SOLUTION
        self.index += 1
        if self.index >= len(self.ordering):
            raise StopIteration
        samples = self.dataset[self.ordering[self.index]]
        return Tensor(samples[0], device = ndl.cpu()), Tensor(samples[1], device = ndl.cpu())
```  
## q3 Convolutional neural network   
1.pad  
axes 是一个列表或元组，表示每个维度需要填充的数量。axes[i] 是一个二元组 (left_pad, right_pad)，表示在第 i 个维度左侧填充 left_pad 个元素，右侧填充 right_pad 个元素。可以先计算出pad之后的shape,构造新矩阵,然后记录下原矩阵应该对应到新矩阵的位置,具体用slice元组实现.  
2.flip  
按照轴翻转数组,实现就是将stride变成负的,将offset变为要翻转的轴的最后一个元素位置,即将翻转维度的最后一个偏移量相加,这种实现方式需要在flip之前调用compact,因为并没有处理offset不为0在数组中间的情况   
3.Dilate  
将每个矩阵元素之间插入pad,整体矩阵扩展,与flip类似,先构建新尺寸的空矩阵
4.卷积前向
5.卷积反向
6.广播op有多维sum,需要手动循环求和,因为没有支持
7.reshape的时候,要先调用compact  

## q4-6 RNN  
问题不大

# hw3
## q1
一个ndarray正常来说,包含以下几个元素  
1.数据存储本身  
2.shape 用于标记矩阵形状,对于切片矩阵,shape不需要与存储本身大小对应  
3.stride 用于矩阵元素位置的计算,即a[i][j] = i * stride[0] + j * stride[1]
- 正常矩阵 stride元组从后往前,分别是shape从后往前的相乘,再乘单个数据所占大小
- 切片矩阵的stride与总的存储数据不完全对应
- 切片矩阵本身的shape和stride计算之后与总的存储数据不完全对应,所以需要进行compact操作,虽然计算可以用,但是进行逐元素操作的时候,会损失数据局部性  
4.广播操作对于stride,只需要把shape中为1的且需要广播的维度stride置为0即可,原理是广播不需要真正对数据进行复制,只是通过修改shape和stride让你感觉在取下标的时候,渠道一样的数据即可,所以广播维度的stride不需要算到索引上

对于这些操作本身来说,除了compact操作之外,其他的操作都不需要进行数据存储层面的修改,只需要对shape,stride,offset修改即可

所以  
1.reshape 需要修改shape,之后再求得新stride使用辅助函数as_stride
2.permute 按照索引轮换对应的shape和stride即可  
3.broadcast_to 首先要保证新shape要么等于原shape,要么是1,之后将不与原shape相等的维度,即为1需要广播的维度的stride,设为0  
4.get_item shape直接拿前后索引差除以step,stride用旧的stride乘上step,offset指的是每个维度最开始的索引,内层的start要加上所有外层的start,是一个规约算法

## q2
1.将一个非紧凑型矩阵转换为紧凑型矩阵,给出了原矩阵, 紧凑后的shape,stride,offset  
构建类似shape的索引向量,类似高精度加法的方式,每次迭代将向量从后往前加1,需要进位.  在根据索引来利用公式a[i][j] = offset+a[i]*stride[i]+a[j]*stride[j]计算出要填入的数值在原矩阵中的位置.对紧凑型矩阵进行for循环即可  
2.将一个紧凑矩阵赋值给一个非紧凑型矩阵,即将1调换位置即可  
3.将标量赋值给紧凑矩阵,类似2
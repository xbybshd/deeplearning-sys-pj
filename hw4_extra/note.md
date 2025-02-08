## q1  
## q2  
因为transformer计算过程存在多维度batch矩阵相乘,所以要对一系列的op进行多batch改造,基本思路都是先把
除最后一个维度之外的维度拍平,进行操作,之后在转回来   
1.matmul 要支持左矩阵为batch  
2.layernorm1d 也要支持
3.linaer的bias需要特判一下,也需要改造一下广播代码,进行多batch改造
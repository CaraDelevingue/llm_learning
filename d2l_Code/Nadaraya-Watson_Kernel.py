import torch

from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

#训练集：0-5之间随机生成50个数
n_train = 50
x_train, _= torch.sort(torch.rand(n_train)*5)

#真实的fx函数
def f(x):
    return 2*torch.sin(x)+x**0.8

#训练集的label：真实函数+方差为0.5的高斯噪音
y_train = f(x_train)+torch.normal(0.0,0.5,(n_train,))
#测试集：0-5之间间隔0.1的均匀点
x_test=torch.arange(0,5,0.1)
#测试集的真实值
y_truth=f(x_test)
n_test=len(x_test)
print(n_test)

#给定x_test预测y_truth
#黄点为训练点
#蓝线为真实的曲线
#粉红线为预测值
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
    plt.show() 
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

 
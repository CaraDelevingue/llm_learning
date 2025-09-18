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
    
#简单预测函数：y_train的平均值

#将平均值重复50（n_test）次
#y_hat = torch.repeat_interleave(y_train.mean(), n_test)
#y_hat = torch.repeat_interleave(y_train.mean(), n_test)



#非参数注意力汇聚（pooling）

# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
#X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))

# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
#attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)

# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
#y_hat = torch.matmul(attention_weights, y_train)



#带参数注意力汇聚（pooling）
'''
#批量矩阵乘法
#创建值全为一的3维向量：ones（x,y,z)=>x个y*z的二维向量
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
#bmm批量矩阵乘法，生成2个1*6的矩阵，返回shape（2，1，6）
torch.bmm(X, Y).shape

#生成大小为2*10，值为0.1的二维矩阵
weights = torch.ones((2, 10)) * 0.1

#arange：生成值为0-19等差数列的一维张量，再变形为2*10的张量，值和顺序不变
values = torch.arange(20.0).reshape((2, 10))
#squeeze（x）:在x位置生成一个维度为1的新轴
#weight.unsqueeze：（2，1，10）
#values.unsqueece：（2，10，1）
#bmm生成（2，1，1）的张量：[[[4.5]],[[14.5]]]
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
'''

#带参数版本的Nadaraya-Watson核回归函数
class NWKernelRegression(nn.Module):
    #**kwargs接受任意关键字参数
    def __init__(self, **kwargs):
        #调用父类初始化：nn.Module初始化
        super().__init__(**kwargs)
        #创建一个[0,1]的随机一维张量作为w，requires_grad表示这个参数需要计算梯度
        #通过Parameter（）将张量包装为神经网络参数
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        #attention_weights的形状为（查询个数，1，键值对个数）
        # values的形状为(查询个数，“键－值”对个数，1)
        #返回的形状为（查询个数，1，1），每个1*1张量为预测值
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
     
#x_train,y_train都为（n_train,）的一维张量   
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
#排除对当前点权重的计算
#e^-0会使当前权重=1非常大，于是w会学习得非常小
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
#损失函数为均方误差损失函数：（1/n)*所有样本的（y_i-y_hat_i)^^2
#reduction='none'返回每个样本的单独损失值，不进行聚合
loss = nn.MSELoss(reduction='none')
#随机梯度下降优化器：用于更新神经网络的参数以最小化损失函数
#net.parameters()获取神经网络中所有需要训练的参数：w
#lr=0.5 学习率为0.5
#每次迭代中：计算梯度，更新参数para=para-lr*grad
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
#可视化工具，实时绘制训练过程中的指标变化
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

#5次训练循环
for epoch in range(5):
    #清空梯度
    trainer.zero_grad()
    #net：神经网络通过query：x、key、value，计算预测值，前向传播
    #loss：预测值和真实值计算损失
    l = loss(net(x_train, keys, values), y_train)
    #sum：累加所有样本的损失
    #backward：计算所有参数的梯度
    l.sum().backward()
    #根据梯度更新参数：para=para-lr*grad
    trainer.step()
    #打印当前epoch和总损失值
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    #将当前epoch的总损失值添加到动画器
    animator.add(epoch + 1, float(l.sum()))

#将测试的query、key、value放入已经训练好（更新后的w）的神经网络中
# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
#squeeze：将一维预测值转化成图表需要的二维输入
#detach：不保留计算图，requires_grad=false
y_hat = net(x_test, keys, values).unsqueeze(1).detach()

plot_kernel_reg(y_hat)

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
 
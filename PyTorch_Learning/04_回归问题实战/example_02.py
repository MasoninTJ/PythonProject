import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import init

true_w = [2, -3.4]
true_b = 4.2


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


m_data = np.genfromtxt('data_02.csv', delimiter=',', dtype=np.float32)
features = torch.from_numpy(m_data[:, :2])
labels = torch.from_numpy(m_data[:, 2])

batch_size = 10
dataset = data.TensorDataset(features, labels)  # 共1000组数据
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)  # 共100个批次的数据，每个批次10个数据，并随机打乱

# 两个输入，一个输出
net = LinearNet(2)
# 这里定义初始值？
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
# 将均方误差损失作为模型的损失函数
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        output = net(X)
        ll = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        ll.backward()
        optimizer.step()

dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)

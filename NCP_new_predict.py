# 基于AI数据模型的疫情预测
'考虑到传染病动力学建模的复杂性和参数确定的复杂度，采用LSTM神经网络对湖北省新增病例数预测，并对比体现防控的意义'

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
'''
# 数据预处理
with open("data.txt", "r", encoding="utf-8") as f:
    data = f.read()
print([data])
data = [row.split(',') for row in data.split("\n")]
value = [int(each[1]) for each in data]
print(data, '\n', value)
# 异常点处理
'''
import pandas as pd
df = pd.read_excel('real_data.xlsx')#这个会直接默认读取到这个Excel的第一个表单
value = df['湖北新增确诊'].values[0:67]
#value = df['全国累计确诊'].values[10:50]
print(len(value))


x = []
y = []
seq = 3
for i in range(len(value)-seq-1):
    x.append(value[i:i+seq])
    y.append(value[i+seq])
#print(x, '\n', y)

train_x = (torch.tensor(x[:50]).float()/1000.).reshape(-1, seq, 1)
train_y = (torch.tensor(y[:50]).float()/1000.).reshape(-1, 1)
test_x = (torch.tensor(x[50:]).float()/1000.).reshape(-1, seq, 1)
test_y = (torch.tensor(y[50:]).float()/1000.).reshape(-1, 1)
#print(train_x, '\n', train_y)
# 模型训练
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.linear = nn.Linear(16 * seq, 1)
    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 16 * seq)
        x = self.linear(x)
        return x

# 模型训练
model = LSTM()
optimzer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_func = nn.MSELoss()
model.train()
l = []
for epoch in range(600):
    output = model(train_x)
    loss = loss_func(output, train_y)
    l.append(loss)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if epoch % 20 == 0:
        tess_loss = loss_func(model(test_x), test_y)
        print("epoch:{}, train_loss:{}, test_loss:{}".format(epoch, loss, tess_loss))

# 模型预测、画图
model.eval()
prediction = list((model(train_x).data.reshape(-1))*1000) + list((model(test_x).data.reshape(-1))*1000)
print(len(value[3:]), len(prediction), len(np.arange(50, 64, 1)), len(prediction[50:64]))
#print('train_x', train_x*1000, '\n', 'train_y', train_y*1000, '\n', 'test_x', test_x*1000, '\n', 'test_y', len(test_y), test_y*1000)
plt.figure(1)
plt.plot(value[3:], label='True Value')
plt.plot(prediction[:51], label='LSTM fit')
plt.plot(np.arange(50, 63, 1), prediction[50:64], label='LSTM pred')
plt.legend(loc='best')
plt.title('New daily infections prediction(Hubei province)')
plt.xlabel('Day')
plt.ylabel('New Confirmed Cases')
#plt.figure(2)
#plt.plot(l)
plt.show()
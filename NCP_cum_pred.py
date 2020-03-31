import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# 数据预处理
import pandas as pd
df = pd.read_excel('real_data.xlsx')
value = df['全国累计确诊'].values[10:67]
print(len(value))
x = []
y = []
seq = 3
for i in range(len(value)-seq-1):
    x.append(value[i:i+seq])
    y.append(value[i+seq])
#print(x, '\n', y)

train_x = (torch.tensor(x[0:40]).float()/100000.).reshape(-1, seq, 1)
train_y = (torch.tensor(y[0:40]).float()/100000.).reshape(-1, 1)
test_x = (torch.tensor(x[40:57]).float()/100000.).reshape(-1, seq, 1)
test_y = (torch.tensor(y[40:57]).float()/100000.).reshape(-1, 1)
print(test_y)
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

for epoch in range(400):
    output = model(train_x)
    loss = loss_func(output, train_y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if epoch % 20 == 0:
        tess_loss = loss_func(model(test_x), test_y)
        print("epoch:{}, train_loss:{}, test_loss:{}".format(epoch, loss, tess_loss))

# 模型预测、画图
model.eval()
prediction = list((model(train_x).data.reshape(-1))*100000) + list((model(test_x).data.reshape(-1))*100000)
plt.plot(value[3:], label='True Value')
plt.plot(prediction[:41], label='LSTM fit')
plt.plot(np.arange(40, 53, 1), prediction[40:], label='LSTM pred')
print(len(value[3:]))
print(len(prediction[40:]))
plt.legend(loc='best')
plt.title('Cumulative infections prediction(mainland China)')
plt.xlabel('Day')
plt.ylabel('Cumulative Cases')
plt.show()

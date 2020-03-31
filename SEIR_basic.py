# SEIR传染病模型仿真
"View more, visit my github repository: https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM"
'没有解析解，ode求解数值解'
'重点是动力学模型的准确性（SEIR还是SEIRS、SIQR、SIQS模型，以及媒体宣传和随机因素的影响），难点是beta，gamma，sigma/Te的取值'
'模型忽略迁入率和迁出率，死亡率，参数设置参考钟院士等文章http://dx.doi.org/10.21037/jtd.2020.02.64'
######################################
# N: 区域内总人口                      #
# S: 易感者                           #
# E: 潜伏者                           #
# I: 感染者                           #
# R: 康复者                           #
# r: 每天接触的人数                    #
# r2: 潜伏者每天接触的人数              #
# beta1: 感染者传染给易感者的概率, I——>S #
# beta2: 潜伏者感染易感者的概率, E——>S   #
# sigma: 潜伏者转化为感染者的概率, E——>I #
# gama: 康复概率, I——>R                #
# T: 传播时间                          #
#######################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
N = 100000          # 湖北省为6000 0000
E_0 = 0
I_0 = 1
R_0 = 0
S_0 = N - E_0 - I_0 - R_0
beta1 = 0.78735     # 真实数据拟合得出
beta2 = 0.15747
# r2 * beta2 = 2
sigma = 0.1         # 1/14, 潜伏期的倒数
gamma = 0.1         # 1/7, 感染期的倒数
r = 1               # 政府干预措施决定
T = 150

#ode求解
INI = [S_0, E_0, I_0, R_0]
def SEIR(inivalue, _):
    X = inivalue
    Y = np.zeros(4)
    # S数量
    Y[0] = - (r * beta1 * X[0] * X[2]) / N - (r * beta2 * X[0] * X[1]) / N
    # E数量
    Y[1] = (r * beta1 * X[0] * X[2]) / N + (r * beta2 * X[0] * X[1]) / N - sigma * X[1]
    # I数量
    Y[2] = sigma * X[1] - gamma * X[2]
    # R数量
    Y[3] = gamma * X[2]
    return Y

T_range = np.arange(0, T+1)
Res = spi.odeint(SEIR, INI, T_range)
S_t = Res[:, 0]
E_t = Res[:, 1]
I_t = Res[:, 2]
R_t = Res[:, 3]

plt.plot(S_t, color='blue', label='Susceptibles')#, marker='.')
plt.plot(E_t, color='grey', label='Exposed')
plt.plot(I_t, color='red', label='Infected')
plt.plot(R_t, color='green', label='Removed')
plt.xlabel('Day')
plt.ylabel('Number')
plt.title('SEIR Model')
plt.legend()
plt.show()
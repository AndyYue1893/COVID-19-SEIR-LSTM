# SEIR传染病模型仿真
"View more, visit my github repository: https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM"
'没有解析解，ode求解数值解'
'重点是动力学模型的准确性（SEIR还是SEIRS、SIQR、SIQS模型，以及媒体宣传和随机因素的影响），难点是beta，gamma，sigma/Te的取值'
'模型忽略迁入率和迁出率，潜伏者传染率beta2，参数设置参考钟院士等文章http://dx.doi.org/10.21037/jtd.2020.02.64'
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

# 阶段一，11.24 - 1.23  1.11
N = 60000000         # 湖北省为6000万，武汉900万
E_0 = 0
I_0 = 1
R_0 = 0
S_0 = N - E_0 - I_0 - R_0
b1 = 0.02         # 真实数据拟合得出
b2 = 0.021/3#0.007
# r2 * beta2 = 2
sigma = 1/14         # 1/14, 潜伏期的倒数
gamma = 1/7          # 1/7, 感染期的倒数
r = 18                # 政府干预措施决定
T = 84

#ode求解
INI = [S_0, E_0, I_0, R_0]
def SEIR(inivalue, _):
    X = inivalue
    Y = np.zeros(4)
    # S数量
    Y[0] = - (r * b1 * X[0] * X[2]) / N - (r * b2 * X[0] * X[1]) / N
    # E数量
    Y[1] = (r * b1 * X[0] * X[2]) / N + (r * b2 * X[0] * X[1]) / N - sigma * X[1]
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
#print(I_t[60], type(I_t)) # 549 <class 'numpy.ndarray'>
#print(E_t[57])


# 阶段二，1.23后
S_2 = S_t[T]
E_2 = E_t[T]
I_2 = I_t[T]
R_2 = R_t[T]

beta1 = 0.02#0.15747     # 真实数据拟合得出
beta2 = 0.021/3#0.78735
# r2 * beta2 = 2
sigma2 = 1/4   # 1/14, 潜伏期的倒数
#gamma = 1/6.736     # 1/7, 感染期的倒数
r2 = 0.1              # 政府干预措施决定
T2 = 150-T

#ode求解
INI = [S_2, E_2, I_2, R_2]
def SEIR(inivalue, _):
    X = inivalue
    Y = np.zeros(4)
    # S数量
    Y[0] = - (r2 * beta1 * X[0] * X[2]) / N - (r2 * beta2 * X[0] * X[1]) / N
    # E数量
    Y[1] = (r2 * beta1 * X[0] * X[2]) / N + (r2 * beta2 * X[0] * X[1]) / N - sigma2 * X[1]
    # I数量
    Y[2] = sigma2 * X[1] - gamma * X[2]
    # R数量
    Y[3] = gamma * X[2]
    return Y

T_range = np.arange(0, T2+1)
Res = spi.odeint(SEIR, INI, T_range)
S_t2 = Res[:, 0]
E_t2 = Res[:, 1]
I_t2 = Res[:, 2]
R_t2 = Res[:, 3]

plt.figure(figsize=(10, 6))
#显示日期
import pandas as pd
xs = pd.date_range(start='20191124', periods=T+1, freq='1D')#生成2020-02-11类型的日期数组（）
#print(xs)
xs2 = pd.date_range(start='20200216', periods=T2+1, freq='1D')

#plt.plot(S_t, color='blue', label='Susceptibles')#, marker='.')
plt.plot(xs, E_t, color='grey', label='Exposed', marker='.')
plt.plot(xs2, E_t2, color='grey', label='Exposed Prediction')
plt.plot(xs, I_t, color='red', label='Infected', marker='.')
plt.plot(xs2, I_t2, color='red', label='Infected Prediction')
plt.plot(xs, I_t + R_t, color='green', label='Infected + Removed', marker='.')
plt.plot(xs2, I_t2 + R_t2, color='green', label='Cumulative Infections Prediction')
#plt.plot(np.arange(0, T+1, 1), I_t + R_t, color='green', label='Removed')
#plt.plot(np.arange(T, T+T2+1, 1), I_t2 + R_t2, color='green', label='Infected2')
#plt.xlabel('Date')
plt.ylabel('Number')
plt.title('SEIR Prediction(Hubei province, 10 days later intervention)')
plt.legend()

xs3 = pd.date_range(start='20200202', periods=1, freq='1D')
#plt.plot(xs3, np.arange(1000, 2000, 1000))
plt.annotate(r'$2.2-Intervention$', xy=(xs3, 200), xycoords='data', xytext=(-44, -60), textcoords='offset points',
             fontsize=10, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
#print(I_t2)
xs4 = pd.date_range(start='20200220', periods=1, freq='1D')
plt.annotate(r'$2.20-Peak$', xy=(xs4, 103911), xycoords='data', xytext=(-25, -130), textcoords='offset points',
             fontsize=10, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
#print(R_t2)
xs5 = pd.date_range(start='20200420', periods=1, freq='1D')
plt.annotate(r'4.20-Epidemic-Scale:261963', xy=(xs5, 261963), xycoords='data', xytext=(-70, -60), textcoords='offset points',
             fontsize=10, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
#plt.text(30, 10, r'$This\ is\ the\ some\ text.\ \mu_j\ \sigma_i\ \alpha_t$')
plt.show()
# Novel-Coronavirus-Pneumonia-SEIR-LSTM
本项目实现2019新型冠状病毒肺炎预测，采用经典传染病动力学模型SEIR，通过控制接触率来改变干预程度，体现防控的意义。

另外，采用LSTM神经网络实现一定程度的预测，输入前三天的数据，预测第四天的数据。下一步要增加干预值作为输入，优化输入输出的序列长度，更好地实现预测。

SEIR and LSTM prediction of the epidemics trend of COVID-19.

先上图
![image](https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM/blob/master/SEIR——basic.png)
![image](https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM/blob/master/SEIR_20200123_Intervention.png)![image](https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM/blob/master/SEIR_20200202_Intervention.png)
![image](https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM/blob/master/NCP_active_predict.png)![image](https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM/blob/master/NCP_new_predict.png)![image](https://github.com/AndyYue1893/Novel-Coronavirus-Pneumonia-SEIR-LSTM/blob/master/NCP_cum_pred.png)

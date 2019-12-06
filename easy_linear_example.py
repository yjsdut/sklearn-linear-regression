import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 产生数据
x = np.linspace(0,10,50)  # 在规定的时间内，返回固定间隔的数据。他将返回“num”个等间距的样本
print(x.shape)  # 0到10等间隔产生50个数 shape:(50,) 一维向量
noise = np.random.uniform(-2, 2, size=50)
y=5*x+noise

# 创建模型
linear =LinearRegression()
# 拟合模型
linear.fit(np.reshape(x,(-1,1)),np.reshape(y,(-1,1)))
print(linear)
# 预测
y_pred =linear.predict(np.reshape(x,(-1,1)))
plt.figure(figsize=(5,5))  # 产生一个窗口
plt.scatter(x,y)  # 画散点图
plt.plot(x,y_pred,color='red')
plt.show()
print(linear.coef_)
print(linear.intercept_)

'''
np.linspace : https://blog.csdn.net/weixin_39010770/article/details/86015822
np.random.uniform：https://blog.csdn.net/u013920434/article/details/52507173
'''
'''
双变量线性回归
'''

import pandas as pd
dataset=pd.read_csv('dataset.csv')
# print(dataset.shape)  # (8,3)
import numpy as np
X=np.asarray(dataset.get(["room_num","area"]))  # (8, 2)
y=dataset.get("price")
# print(X.shape)

# 划分数据
X_train = X[:-2]  # 第1维一共8个数据，取走5个作为训练
X_test = X[-2:]  # 最后3个作为test
y_train = y[:-2]
y_test = y[-2:]  # 标签

from sklearn import  linear_model
regr=linear_model.LinearRegression().fit(X_train,y_train)  # 拟合模型，regr是一个实例化的对象，可以调用对象的各种函数
y_pred = regr.predict(X_test)

print("Coefficients",regr.coef_)  # 模型权重
print("Intercept",regr.intercept_)  # 模型偏置
print("the model is : y = ",regr.coef_,'* X + ',regr.intercept_)

# 均方误差
from sklearn.metrics import mean_squared_error,r2_score
print('Mean squared error: %.2f'%mean_squared_error(y_test,y_pred))
print('Variance score: %.2f'%r2_score(y_test,y_pred))  # r2_score越接近1越好

# 画图
# 这一段是为了画模型准备的数据，因为模型是平面，所以需要很多个点才能画出
# 给room_num 和area，用coef_[0]*room +coef_[1]*area + intercept_就得到一个点，
# 然后我们需要很多点，才能画一个平面，所以利用np.meshgrid扩增了输入数据room_num 和area，
# 然后都按照上面的操作进行，最终的到y，然后画很多点，得到了平面
x0,x1 = np.meshgrid(np.asarray(X_train)[:,0],np.asarray(X_train)[:,1])  # 画三维图的时候要转为meshgrid，然后后米纳在画图
print(x0)
print(np.asarray(X_train)[:,0])  # [2 3 2 3 2 2]
print(x1)
print(np.asarray(X_train)[:,1])  # [100  60 110  50  90  80]
y = np.asarray(regr.coef_[0]*x0+regr.coef_[1]*x1+regr.intercept_)
y_pred = regr.predict(X_train)
print(y)
print(y_pred)  # 只有(1,1) (2,2) (3,3)位置的点，才是原数据得到的点，其他位置都是我们自己为了画平面扩增的点

import matplotlib.pyplot as plt
fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
ax.set_xlabel("room_num")
ax.set_ylabel("area")
ax.set_zlabel("price")

# # 画训练集的散点图
ax.scatter(np.asarray(X_train)[:, 0], np.asarray(X_train)[:, 1], np.asarray(y_train), alpha=0.8, color='black')
# 画模型，三维空间中的一个平面
ax.plot_surface(x0, x1, y, shade=False)
plt.show()
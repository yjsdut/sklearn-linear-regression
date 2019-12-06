'''
单变量线性回归，使用sklearn
'''
# 准备数据集
import pandas as pd
dataset = pd.read_csv('dataset.csv')
# print(dataset.shape)   # (8, 3)
import numpy as np
X = np.asarray(dataset.get('area')).reshape(-1,1)  # (8, 1) X变成了tensor，维度是2
y=dataset.get('price')  # shape(8,)
# print(X)  # X经过了np，是向量
# print(y[0])  # y是列表形式的数字 y[0]  y[1].....形式,每个值是int类型的数字y[0]+y[1]，可执行加减等运算

# 划分数据
X_train = X[:-3]  # 第1维一共8个数据，取走5个作为训练
X_test = X[-3:] # 最后3个作为test
y_train = y[:-3]
y_test = y[-3:]  # 标签

from sklearn import  linear_model
regr=linear_model.LinearRegression().fit(X_train,y_train)  # 拟合模型，regr是一个实例化的对象，可以调用对象的各种函数
y_pred = regr.predict(X_test)

# 输出模型的参数：https://www.jianshu.com/p/6a818b53a37e
print("Coefficients",regr.coef_)  # 模型权重
print("Intercept",regr.intercept_)  # 模型偏置
print("the model is : y = ",regr.coef_,'* X + ',regr.intercept_)

# 均方误差
from sklearn.metrics import mean_squared_error,r2_score
print('Mean squared error: %.2f'%mean_squared_error(y_test,y_pred))
print('Variance score: %.2f'%r2_score(y_test,y_pred))  # r2_score越接近1越好

# 画图
import matplotlib.pyplot as plt
plt.xlabel('area')
plt.ylabel('price')
# 画训练集的散点图
plt.scatter(X_train,y_train,alpha=0.8,color='black')
# 画模型
plt.plot(X_train,regr.coef_*X_train+regr.intercept_,color='red',linewidth=1) #  这句话和下面那一句一样
plt.plot(X_train,regr.predict(X_train),color='blue',linewidth=1)
plt.show()


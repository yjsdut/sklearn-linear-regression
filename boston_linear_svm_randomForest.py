'''
波士顿房价预测案例：https://blog.csdn.net/csdn_fzs/article/details/82351821
简单使用几种回归器
'''

# 加载数据
from sklearn import datasets
boston_data =datasets.load_boston()
x_full = boston_data.data  # 加载load_boston()中所有数据，一共有506条记录，13条特征（变量）
y = boston_data.target  # 加载load_boston()中的目标值，即标签
# print(x_full.shape)  # (506, 13)  506个数据，每个数据有13个特征
# print(y.shape)   # (506,)

# https://github.com/yjsdut/sklearn-linear-regression.git

# 转换和分析
from sklearn.feature_selection import SelectKBest,f_regression
selector = SelectKBest(f_regression,k=1).fit(x_full,y)  # 选出相关性最强的SelectKBest类作为特征
#  SelectKBest特征选择 : https://www.jianshu.com/p/b3056d10a20f
# f_regression：相关系数，计算每个变量与目标变量的相关系数，然后计算出F值和P值
# SelectKBest是一个类，selector是这个类的一个实例化对象,对象没有shape属性，他不是向量，是对象
# 为每一行数据，13个特征中选出和标签相关性最强的特征的值
x = x_full[:,selector.get_support()]  # 采用get_support()将数据缩减成一个向量，即数据降维
# print(x.shape)    # (506, 1)

# 可视化
import matplotlib.pyplot as plt
def plot_scatter(x,y,R=None):
    plt.scatter(x,y,s=32,marker='o',facecolors='blue')
    if R is not None:
        plt.scatter(x,R,color='red',linewidths=0.5)
    plt.show()

# 画特征和标签的图
plot_scatter(x,y)

# 线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True).fit(x,y)
plot_scatter(x,y,regressor.predict(x))

# 支持向量机 SVM回归模型
from sklearn.svm import SVR
regressor = SVR().fit(x,y)
plot_scatter(x,y,regressor.predict(x))

# 使用随机森林回归模型
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor().fit(x,y)
plot_scatter(x,y,regressor.predict(x))
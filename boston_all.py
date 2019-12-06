'''
boston 房价预测整体,boston数据集有506个数据，每个数据有13个特征，数据有data部分和target部分,获取的数据都是float类型的
python3
'''
import numpy as np
from sklearn import datasets
boston = datasets.load_boston()
# print(boston.DESCR)  # 打印描述文档 506个数据，每个数据有13个特征

# 划分train test 利用采样频率划分
sampleRatio = 0.5
n_samples = len(boston.target)  # 506
# print(n_samples)
sampleBoundary = int(sampleRatio * n_samples)  # 训练集个数
# 洗乱整个集合，并取出相应的train 和test 数据集
shuffleIdx = list(range(n_samples))  # 获取数据编号
np.random.shuffle(shuffleIdx)  # 打乱
# train的特征和回归值
train_features = boston.data[shuffleIdx[:sampleBoundary]]  # shape(253,13)  取出打乱以后的前一半作为train
# print(train_features.shape)
train_target = boston.target[shuffleIdx[:sampleBoundary]]
#test 的特征和回归值
test_features = boston.data[shuffleIdx[sampleBoundary:]]
test_target = boston.target[shuffleIdx[sampleBoundary:]]

# 获取回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # 获取模型
regressor.fit(train_features,train_target)  # 拟合
test_pred = regressor.predict(test_features)  # 得到test的预测结果

# 画出预测结果的图
import matplotlib.pyplot as plt
plt.plot(test_pred,test_target,'rx')  # 画出test的预测值和test的真实值
# 表示，横坐标为预测值，纵坐标为真实值，如果预测值和真实值相同，就会显示在图中的y=x这条线上，
# 也就是，y=x这条线上的值，预测准确
plt.plot([test_pred.min(),test_pred.max()],[test_pred.min(),test_pred.max()],'b-',lw=4)
plt.xlabel("预测值")
plt.ylabel("真实值")
plt.show()
# 蓝色线上的点准确，蓝色上下的点错误


########################################################################################
# # 交叉验证
# # 基本思想就是将原始数据（dataset）进行分组，一部分做为训练集来训练模型，另一部分做为测试集来评价模型。
# # 交叉验证的意义：https://blog.csdn.net/qq_39751437/article/details/85071395
# # 交叉验证的意义：https://blog.csdn.net/u010167269/article/details/51340070
# from sklearn import datasets
# from sklearn.model_selection import  cross_val_predict   # 模型选择库
# from sklearn import linear_model
# from sklearn import datasets
#
# import matplotlib.pyplot as plt
# regressor = linear_model.LinearRegression()
# boston = datasets.load_boston()
# y = boston.target
#
# predicted = cross_val_predict(regressor,boston.data,y,cv=10)
# # cross_val_predict ： https://www.cnblogs.com/lzhc/p/9175707.html
# # 用来评价线性回归器用来房价回归到底好还是不好，而不是为了从10个线性回归器中选择一个好的线性回归器
# # 采用的是k-fold cross validation的方法
# # print(predicted.shape)  # (506,)
# fig,ax =plt.subplots()  # fig, axes = plt.subplots(23)：即表示一次性在figure上创建成2*3的网格，使用plt.subplot()只能一个一个的添加
# ax.scatter(y,predicted,edgecolors=(0,0,0,))
# ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=4)
# ax.set_xlabel("measured")
# ax.set_ylabel("predicted")
# plt.show()
# # 交叉验证模型打分
# # from sklearn import cross_validation train_test_split
# from sklearn.model_selection import cross_val_score
# print (cross_val_score(regressor,boston.data,y,cv=10))
# # 采用的是k-fold cross validation的方法
# # 默认算的就是r-squared系数  值为多少好？https://blog.csdn.net/weixin_43715458/article/details/94405046
# # R2：反映的是，模型本身对于自变量和因变量的反映，而损失函数是反映两种模型的好坏
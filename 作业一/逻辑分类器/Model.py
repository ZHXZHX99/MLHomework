"""
作业要求：实现10个简单的逻辑分类器识别手写数字
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    h = 1./(1+np.exp(-z))
    return h
    
def de_sigmoid(z,h):
    return h*(1-h)

def initialParm(dim):
    w = np.random.randn(dim,1)
    b = 0
    return w,b

# 标签处理函数——将待识别的手写数字的标签置1，其余为0
def DataClass(Y, num_class):
    Iten = Y.shape[0]   #50000*1
    for i in range(Iten):
        if(Y[i] == num_class):
            Y[i] = 1
        else:
            Y[i] = 0
    return Y

# 前向传播函数——计算目标函数及损失函数
def f_propagate(w, b, X):
    # 获取样本数m：
    m = X.shape[1]
    # 计算目标函数
    A = sigmoid(np.dot(w.T,X)+b)

    return A

# 反向传播函数——梯度下降法 更新权重及偏置
def b_propagate(A, X, Y):
    # 获取样本数m：
    m = X.shape[1]             
    # 计算损失函数
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m
    # 更新权重偏置
    dw = (np.dot(X,(A-Y).T))/m
    db = (np.sum((A-Y)))/m

    return dw, db, cost

# 优化函数——对目标函数进行梯度下降式的有限次迭代计算并将Cost结果可视化
def GD_optimize(w, b, X, Y, num_iterations, learning_rate, num_class):
    # 存储不同阶段cost，观察cost下降情况
    costs = []
    # 进行迭代：
    for i in range(num_iterations):
        # 用propagate计算出每次迭代后的cost和梯度：
        # 前向传播计算目标函数
        A = f_propagate(w,b,X)
        # 反向传播计算梯度
        dw, db, cost = b_propagate(A,X,Y)
        # 利用梯度来更新参数：
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # 每20次迭代，保存一个cost看看：
        if i % 20 == 0:
            costs.append(cost)

        # 打印cost
        if i % 20 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    # # 绘制损失函数值的图像
    # x = range(len(costs))
    # plt.plot(x,costs)
    # plt.xlabel(r'per 20 iterations')
    # plt.ylabel(r'Costs')
    # plt.title("LR model for Number "+ str(num_class))
    # plt.show()

    return w, b

# 预测函数
def predict(w,b,X):
    m = X.shape[1]
    Y_hat = np.zeros((1,m))
    # 计算目标函数
    A = sigmoid(np.dot(w.T,X)+b)
    # 根据sigmoid函数的特性激活
    for  i in range(m):
        if A[0,i]>0.5:
            Y_hat[0,i] = 1
        else:
            Y_hat[0,i] = 0

    return Y_hat

# 模型文件
def logistic_model(X_train,Y_train,X_test,Y_test,learning_rate,num_iterations,num_class):
    # 初始化参数：
    dim = X_train.shape[0]
    W,b = initialParm(dim)

    # 训练优化过程
    W, b = GD_optimize(W,b,X_train,Y_train,num_iterations,learning_rate,num_class)

    # 预测过程
    prediction_train = predict(W,b,X_train)
    prediction_test = predict(W,b,X_test)

    # 计算准确率
    accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    accuracy_test = 1 - np.mean(np.abs(prediction_test - Y_test))
    print("Accuracy on train set:",accuracy_train )
    print("Accuracy on test set:",accuracy_test )

    return accuracy_train, accuracy_test


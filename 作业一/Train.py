import pickle as cPickle
import gzip
import numpy as np
from LRModel import *
from BPmodel import *


# 载入MNIST数据集
def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='latin1')
    f.close()
    # 解析数据  将验证集数据与训练集数据合并训练
    training_inputs = np.concatenate((training_data[0],validation_data[0]),axis=0)   
    training_labels= np.concatenate((training_data[1],validation_data[1]),axis=0)   

    test_inputs =  test_data[0]
    test_labels= test_data[1]

    return (training_inputs,training_labels, test_inputs,test_labels)

# 构造one-hot向量
def vectorized_label(N,labs):
    # 构造One-hot向量表示的标签
    onehot_labs = np.zeros([N, 10])
    for i in range(N):
        id = int(labs[i])
        onehot_labs[i, id] = 1
    return onehot_labs

# 运行模型
if __name__=="__main__":
    # 加载训练数据
    training_inputs,training_labels, test_inputs,test_labels = load_data()
    
    """
    三层BP神经网络
    """
    # training_labels = vectorized_label((list(training_inputs.shape)[0]),training_labels)
    # test_labels = vectorized_label((list(test_inputs.shape)[0]),test_labels)    
    # N, D = np.shape(training_inputs)
    # # 定义网络结构
    # num_hidden = [192, 30, 10]
    # act_funs = [sigmod, sigmod, no_active]
    # de_act_funs = [de_sigmoid, de_sigmoid, de_no_active]

    # # 定义超参数
    # n_epoch = 500
    # batchsize = 100
    # learn_rate = 0.01
    # # BP网络训练并测试
    # model = BPNet(training_inputs, training_labels, test_inputs, test_labels, num_hidden, act_funs, de_act_funs, n_epoch, batchsize, learn_rate)

    """
    10个逻辑二分类器
    """
    res = np.zeros((2,2))

    """"
    数字”0“的二分类器
    """
    # 数字”0“的标签处理
    training_labels_0 = DataClass(training_labels,0)
    test_labels_0 = DataClass(test_labels,0)
    # 训练数字“0”的逻辑回归分类器，并测试其精度
    res[0][0],res[0][1]  = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 3000, learning_rate = 0.005, num_class = 0)

    """"
    数字”1“的二分类器
    """
    # 数字”1“的标签处理
    training_labels_1 = DataClass(training_labels,1)
    test_labels_1 = DataClass(test_labels,1)
    # 训练数字“1”的逻辑回归分类器，并测试其精度
    res[1][0],res[1][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 3000, learning_rate = 0.005, num_class = 1)
    
    # """"
    # 数字”2“的二分类器
    # """
    # # 数字”2“的标签处理
    # training_labels_2 = DataClass(training_labels,2)
    # test_labels_2 = DataClass(test_labels,2)
    # # 训练数字“2”的逻辑回归分类器，并测试其精度
    # res[2][0], res[2][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 2)
    
    # """"
    # 数字”3“的二分类器
    # """
    # # 数字”3“的标签处理
    # training_labels_3 = DataClass(training_labels,3)
    # test_labels_3 = DataClass(test_labels,3)
    # # 训练数字“3”的逻辑回归分类器，并测试其精度
    # res[3][0], res[3][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 3)
    
    # """"
    # 数字”4“的二分类器
    # """
    # # 数字”4“的标签处理
    # training_labels_4 = DataClass(training_labels,4)
    # test_labels_4 = DataClass(test_labels,4)
    # # 训练数字“4”的逻辑回归分类器，并测试其精度
    # res[4][0], res[4][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 4)
    
    # """"
    # 数字”5“的二分类器
    # """
    # # 数字”5“的标签处理
    # training_labels_5 = DataClass(training_labels,5)
    # test_labels_5 = DataClass(test_labels,5)
    # # 训练数字“5”的逻辑回归分类器，并测试其精度
    # res[5][0], res[5][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 5)
    
    # """"
    # 数字”6“的二分类器
    # """
    # # 数字”6“的标签处理
    # training_labels_6 = DataClass(training_labels,6)
    # test_labels_6 = DataClass(test_labels,6)
    # # 训练数字“6”的逻辑回归分类器，并测试其精度
    # res[6][0], res[6][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 6)
    
    # """"
    # 数字”7“的二分类器
    # """
    # # 数字”7“的标签处理
    # training_labels_7 = DataClass(training_labels,7)
    # test_labels_7 = DataClass(test_labels,7)
    # # 训练数字“7”的逻辑回归分类器，并测试其精度
    # res[7][0], res[7][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 7)
    
    # """"
    # 数字”8“的二分类器
    # """
    # # 数字”8“的标签处理
    # training_labels_8 = DataClass(training_labels,8)
    # test_labels_8 = DataClass(test_labels,8)
    # # 训练数字“8”的逻辑回归分类器，并测试其精度
    # res[8][0], res[8][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 8)
    
    # """"
    # 数字”9“的二分类器
    # """
    # # 数字”9“的标签处理
    # training_labels_9 = DataClass(training_labels,9)
    # test_labels_9 = DataClass(test_labels,9)
    # # 训练数字“9”的逻辑回归分类器，并测试其精度
    # res[9][0], res[9][1] = logistic_model(training_inputs.T, training_labels.T, test_inputs.T, test_labels.T, num_iterations = 2000, learning_rate = 0.005, num_class = 9)

    # np.savetxt("result.txt", res)

    # # 绘制数字精度的图像
    # x = range(10)
    # y = res[:,1]
    # plt.plot(x,y)
    # plt.xlabel(r'Number to be recognized')
    # plt.ylabel(r'Accuracy on test data')
    # plt.title("Test Accuracy for 10 numbers")
    # plt.show()

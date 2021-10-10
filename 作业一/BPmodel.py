'''
关于实现BP神经网络：
    作业要求的3层BP网络结构： 784-192-30-10
'''
import numpy as np
import matplotlib.pyplot as plt
# 首先定义需要用到的函数
# sigmoid
def sigmod(z):
    h = 1. / (1 + np.exp(-z))
    return h
# sigmoid导数
def de_sigmoid(z, h):
    return h * (1 - h)
# relu函数

# 前向传播时的无激活函数
def no_active(z):
    h = z
    return h
# 反向传播时无激活函数
def de_no_active(z, h):
    return np.ones(h.shape)

# 交叉熵的前向传播计算
def loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    loss_ce = np.sum(-lab * np.log(p))
    return loss_ce
# 交叉熵的反向传播计算
def de_loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    return p - lab


# 初始化网络结构
def initialNet(dim_in, num_hidden, act_funs, de_act_funs):
    """
    :param dim_in: dim_in:输入层的特征维度
    :param num_hidden: 隐藏层节点数
    :param act_funs: 每层的激活函数
    :param de_act_funs: 反向传播时的函数
    :return: layers：网络层级结构
    """
    layers = []
    # 逐层的进行网络构建
    for i in range(len(num_hidden)):
        layer = {}
        # 定义每一层的权重
        if i == 0: #如果是输入层
            layer["w"]= 0.2*np.random.randn(dim_in,num_hidden[i])-0.1 # 用sigmoid激活函数
        else:
            layer["w"]= 0.2*np.random.randn(num_hidden[i-1],num_hidden[i])-0.1 # 用sigmoid激活函数
        # 定义每一层的偏置
        layer["b"] = 0.1 * np.ones([1, num_hidden[i]])
        layer["act_fun"] = act_funs[i]
        layer["de_act_fun"] = de_act_funs[i]
        layers.append(layer)
    return layers

# 前向传播函数
# 返回每一层的输入与最后一层的输出
def f_forward(datas, layers):
    input_h = []
    input_z = []
    for i in range(len(layers)):
        layer = layers[i]
        if i == 0:
            inputs = datas
            z = np.dot(inputs, layer["w"]) + layer["b"]
            out = layer['act_fun'](z)
            input_h.append(inputs)
            input_z.append(z)
        else:
            inputs = out
            z = np.dot(inputs, layer["w"]) + layer["b"]
            out = layer['act_fun'](z)
            input_h.append(inputs)
            input_z.append(z)
    return input_h, input_z, out

# 反向传播方法的函数
# 进行参数更新更新
def optimize(datas, labs, layers, learn_rate):
    N, D = np.shape(datas)
    # 进行前馈操作
    input_h, input_z, output = f_forward(datas, layers)
    # 计算 loss
    loss = loss_CE(output, labs)
    # 从后向前计算
    deltas0 = de_loss_CE(output, labs)
    # 从后向前计算误差
    deltas = []
    for i in range(len(layers)):
        index = -i - 1
        if i == 0:
            h = output
            z = input_z[index]
            delta = deltas0 * layers[index]["de_act_fun"](z, h)
        else:
            h = input_h[index + 1]
            z = input_z[index]
            # print(layers[index]["de_act_fun"](z,h)[1])
            delta = np.dot(delta, layers[index + 1]["w"].T) * layers[index]["de_act_fun"](z, h)

        deltas.insert(0, delta)

    # 利用误差 对每一层的权重进行修成
    for i in range(len(layers)):
        # 计算 dw 与 db
        dw = np.dot(input_h[i].T, deltas[i])
        db = np.sum(deltas[i], axis=0, keepdims=True)
        # 梯度下降
        layers[i]["w"] = layers[i]["w"] - learn_rate * dw
        layers[i]["b"] = layers[i]["b"] - learn_rate * db

    return layers, loss

# 测试精度
def test_accuracy(datas, labs_true, layers):
    _, _, output = f_forward(datas, layers)
    lab_det = np.argmax(output, axis=1)
    labs_true = np.argmax(labs_true, axis=1)
    N_error = np.where(np.abs(labs_true - lab_det) > 0)[0].shape[0]

    error_rate = N_error / np.shape(datas)[0]
    return error_rate

# 构建BP网络进行训练
def BPNet(training_inputs, training_labels, test_inputs, test_labels, num_hidden, act_funs, de_act_funs, n_epoch, batchsize, learn_rate):
    N,D = training_inputs.shape
    # 初始化网络
    layers = initialNet(D,num_hidden,act_funs,de_act_funs)
    lossres = []
    # 前向传播+反向传播进行训练
    N_batch = N // batchsize
    for i in range(n_epoch):
        # 数据打乱
        rand_index = np.random.permutation(N).tolist()
        # 每个batch 更新一下weight
        loss_sum = 0
        for j in range(N_batch):
            index = rand_index[j * batchsize:(j + 1) * batchsize]
            batch_inputs = training_inputs[index]
            batch_labs = training_labels[index]
            # 分batch进行梯度下降
            layers, loss = optimize(batch_inputs, batch_labs, layers, learn_rate)
            loss_sum = loss_sum + loss
        # 打印每个epoch的损失
        lossres.append(loss_sum)
        print("epoch %d    loss_all %.2f" % (i, loss_sum))
        # 可视化损失
    x = range(len(lossres))
    plt.plot(x,lossres)
    plt.xlabel(r'Number of epoch')
    plt.ylabel(r'Loss on train')
    plt.title("Test Accuracy for 10 numbers")
    plt.show()
    # 测试精度
    error = test_accuracy(test_inputs, test_labels, layers)
    # 打印结果
    print("Accuarcy on Test Data %.2f %%" % ((1 - error) * 100))
    return layers
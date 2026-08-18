import numpy as np

def feed_forward(inputs,outputs,weights):

    pre_hidden = np.dot(inputs,weights[0])+weights[1]
    #在隐藏层上使用sigmoid函数
    hidden = 1/(1+np.exp(-pre_hidden))
    #通过对隐藏层激活值和权重，计算出输出层，并加上偏置
    pred_out = np.dot(hidden,weights[2])+weights[3]
    #计算均方差
    mean_squared_error = np.mean(np.square(pred_out - outputs))

    return mean_squared_error


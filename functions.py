# 所有使用的包
import numpy as np
import struct
import matplotlib.pyplot as plt




# 1.激活函数
def softmax(x):
    M = np.max(x, axis=0, keepdims = True)
    x = x-M
    epsilon = 1e-8
    s = np.exp(x) / (epsilon+np.sum(np.exp(x), axis=0, keepdims=True))
    return s

def relu(x):
    s = np.maximum(0,x)
    return s

def forward_propagation(W1,b1,W2,b2,data):
    hiddenlayer_output = relu(np.matmul(data, W1) + b1)
    outlayer =  relu(np.matmul(hiddenlayer_output, W2) + b2)
    scores = np.exp(outlayer)  # self.batchsize * 10
    scores_sum = np.sum(scores, axis=1, keepdims=True)  # self.batchsize * 1
    return hiddenlayer_output,outlayer,scores,scores_sum


#2.反向传播和梯求计算
def backward_propagation(batchsize,labels,outlayer,hiddenlayer_output,data,W2,W1,reg_factor):
    
    scores = np.exp(outlayer)  # self.batchsize * 10
    scores_sum = np.sum(scores, axis=1, keepdims=True)  # self.batchsize * 1
    res = scores / scores_sum  # self.batchsize * 10
    for i in range(batchsize):
        res[i][int(labels[i])] -= 1
    res  /= batchsize
    dz2=res
    dW2 = np.matmul( hiddenlayer_output.T, dz2)  # self.hiddensize * 10
    db2 = np.sum(dz2, axis=0, keepdims=True) # 1 * 10 
    da1 = np.dot( dz2, W2.T)
    dz1 = np.multiply(da1, np.int64(hiddenlayer_output > 0))
    dW1 = np.dot( data.T, dz1)
    db1 = np.sum( dz1, axis=0, keepdims=True)


    # L2正则化求导项
    dW2 += reg_factor * W2 /batchsize
    dW1 += reg_factor * W1 /batchsize

    return dW1,dW2,db1,db2
#3 计算损失
def compute_cost(W1,W2,batchsize,labels,reg_factor,outlayer):
    epsilon = 1e-8
    logprobs = np.multiply(labels,-np.log(outlayer+epsilon))
    cost = 1./batchsize * np.sum(logprobs)
    cost = cost+(np.sum(np.square(W1))+np.sum(np.square(W2)))*reg_factor/(2*batchsize)
    return cost

def compute_test_cost(testdata,W1,W2,b1,b2,testlabel,reg_factor):
    hiddenlayer_output = relu(np.matmul(testdata, W1) + b1)
    outlayer = relu(np.matmul(hiddenlayer_output, W2) + b2)
    epsilon = 1e-8
    logprobs = np.multiply(testlabel,-np.log(outlayer+epsilon))
    cost = 1./10000 * np.sum(logprobs)
    cost = cost+(np.sum(np.square(W1))+np.sum(np.square(W2)))*reg_factor/(2*10000)
    return cost

#4 更新参数
def update_parameters_with_sgd(W2,dW2,W1,dW1,b2,db2,b1,db1,learning_rate):
    W2 += -learning_rate * dW2
    W1 += -learning_rate * dW1
    b2 += -learning_rate * db2
    b1 += -learning_rate * db1
    return W2,W1,b2,b1

#5 向前传播
def visualization_acc(trainaccuracy,testaccuracy,epoch):
    plt.plot(range(1,epoch),trainaccuracy,label='train')
    plt.plot(range(1,epoch),testaccuracy,label='test')
    plt.xlabel('epoch num')
    plt.ylabel('accuracy ')
    plt.title('training & testing accuary')
    plt.legend()
    plt.show()
    

    
    
def visualization_loss(trainloss,testloss,epoch):
    plt.plot(range(1,epoch),trainloss,label='train')
    plt.plot(range(1,epoch),testloss,label='test')
    plt.xlabel('epoch num')
    plt.ylabel('loss ')
    plt.title('training & testing loss')
    plt.legend()
    plt.show()
    

    plt.show()

def layers_loss(x,y):
    plt.scatter(x, y)
    plt.xlabel('lr/L2 wight/hiddenlayersize')
    plt.ylabel('accuracy ')
    plt.title('lr/L2 wight/hiddenlayersize vs accuracy')
    plt.legend()
    plt.xticks(rotation=90) 
    plt.show()
# 定义解析文件读取图像数据的函数
def images_data(dtype):
    if dtype == 'train':
        data = open('./data/MNIST/raw/train-images-idx3-ubyte', 'rb').read()
    else:
        data = open('./data/MNIST/raw/t10k-images-idx3-ubyte', 'rb').read()
    index = 0
    # 文件头信息：魔数、图片数、每张图高、图宽
    # 32位整型采用I格式
    fmt_header = '>IIII'
    magicnum, imagenum, rownum, colnum = struct.unpack_from(fmt_header, data, index)
    # 数据在缓存中的指针位置 index此时为16
    index += struct.calcsize('>IIII')  

    output = np.empty((imagenum, rownum * colnum))
    # 图像数据像素值类型Unsigned char型(B)同时大小为28*28 784
    fmt_image = '>' + str(rownum * colnum) + 'B'
    for i in range(imagenum):
        output[i] = np.array(struct.unpack_from(fmt_image, data, index)).reshape((1, rownum * colnum))/255.
        index += struct.calcsize(fmt_image)
    mu = np.mean(output, axis=1, keepdims=True)
    sigma = np.std(output, axis=1, keepdims=True)
    return (output - mu)/sigma



# 定义解析文件读取标签数据的函数
def labels_data(dtype):
    if dtype == 'train':
        data = open('./data/MNIST/raw/train-labels-idx1-ubyte', 'rb').read()
    else:
        data = open('./data/MNIST/raw/t10k-labels-idx1-ubyte', 'rb').read()
    index = 0
    # 文件头信息：魔数、标签数也就是数据量
    fmt_header = '>II'
    magicnum, labelnum = struct.unpack_from(fmt_header, data, index)

    index += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty((labelnum, 1))
    for i in range(labelnum):
        labels[i] = np.array(struct.unpack_from(fmt_image, data, index)[0]).reshape((1, 1))
        index += struct.calcsize(fmt_image)
    return labels



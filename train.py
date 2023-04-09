from functions import *
from parameter import *

class Sol:
    def __init__(self, traindata, testdata, testlabel, learning_rate, regularization_factor, sizeofhiddenlayer,batch_size):
        self.datanum = len(traindata)  # 训练集个数
        self.traindata = traindata  # 训练集数据
        self.testdata = testdata
        self.testlabel = testlabel
        self.lrate = learning_rate  # 学习率
        self.startlrate = learning_rate
        self.reg_factor = regularization_factor  # 正则化强度
        self.sizeofhidden = sizeofhiddenlayer  # 隐藏层神经元个数
        self.batchsize = batch_size
        self.loss = 0
        self.loss_list=[]
        self.test_loss_list=[]
        self.trainaccuracy=[]
        self.testaccuracy=[]
        self.epoch=5

        # 初始化网络(两层) 0-9 10个手写数字类别
        self.W1 = (np.random.rand(28 * 28, self.sizeofhidden) - 0.5) * 2/ 28
        self.b1 = np.zeros((1, self.sizeofhidden))
        self.W2 = (np.random.rand(self.sizeofhidden, 10) - 0.5) * 2 / np.sqrt(self.sizeofhidden)
        self.b2 = np.zeros((1, 10))

        self.allW1 = [self.W1]
        self.allW2 = [self.W2]
        self.allb1 = [self.b1]
        self.allb2 = [self.b2]
    
    def train(self, epoch):
        self.epoch=epoch
        iternum = self.datanum//self.batchsize
        for _ in range(epoch):
            np.random.shuffle(self.traindata)
            self.loss=0
            for i in range(iternum):
                self.lrate = self.lrate * 0.999
                images = self.traindata[i * self.batchsize: (i+1) * self.batchsize, :-1]
                labels = self.traindata[i * self.batchsize: (i+1) * self.batchsize, -1:]
                #self.trainacc.append(self.predict(images, labels))
                self.updateparamters(images, labels)
            self.loss_list.append(self.loss/iternum)
            self.test_loss_list.append(np.mean(compute_test_cost(self.testdata,self.W1,self.W2,self.b1,self.b2,self.testlabel,self.reg_factor)))
            #print('### epoch ' + str(_+1) + ' Done ###')
            self.trainaccuracy.append(self.predict(images, labels))
            self.testaccuracy.append(self.predict(self.testdata, self.testlabel))

    
    def updateparamters(self, data, labels):
        # 激活函数 ReLu
        hiddenlayer_output = relu(np.matmul(data, self.W1) + self.b1)
        outlayer = relu(np.matmul(hiddenlayer_output, self.W2) + self.b2)
        # 损失函数 = 交叉熵损失项 + L2正则化项
        loss = compute_cost(self.W1,self.W2,self.batchsize,labels,self.reg_factor,outlayer)
        self.loss=self.loss+np.mean(loss) # 储存每个batch训练的损失大小
        # 反向传播更新参数

        dW1,dW2,db1,db2=backward_propagation(self.batchsize,labels,outlayer,hiddenlayer_output,data,self.W2,self.W1,self.reg_factor)


        # 更新参数
        self.W2, self.W1,self.b2,self.b1 = update_parameters_with_sgd(self.W2,dW2,self.W1,dW1,self.b2,db2,self.b1,db1,self.lrate)
        self.allW1.append(self.W1)
        self.allW2.append(self.W2)
        self.allb1.append(self.b1)
        self.allb2.append(self.b2)
        
        return
    

    def predict(self, testdata, testlabel):
        hiddenlayer_output = relu(np.matmul(testdata, self.W1) + self.b1)
        outlayer = relu(np.matmul(hiddenlayer_output, self.W2) + self.b2)
        prediction = np.argmax(outlayer, axis=1).reshape((len(testdata),1))
        accuracy = np.mean(prediction == testlabel)
        return accuracy
    

    def savemodel(self):
        paramters = {}
        paramters['W1'] = self.W1
        paramters['W2'] = self.W2
        paramters['b1'] = self.b1
        paramters['b2'] = self.b2
        with open('bestmodel.pkl', 'wb') as f:
            pickle.dump(paramters, f)


def findbest(traindata, testdata, testlabel,batch_size,learningRateList,L2_lambdas_list,sizeofhidden_list,epoch):
    acclist = {}
    plotx= []
    ploty= []
    for lr in learningRateList:
        for regul in L2_lambdas_list:
            for sizeofhidden in sizeofhidden_list:
                a = Sol(traindata, testdata, testlabel, lr, regul, sizeofhidden,batch_size)
                a.train(epoch)
                print(str(lr) + '/' + str(regul) + '/' + str(sizeofhidden) + '模型在测试集准确率为' + str(a.predict(testdata, testlabel)))
                acclist[str(lr) + '/' + str(regul) + '/' + str(sizeofhidden)] = a.predict(testdata, testlabel)
                plotx.append(str(lr) + '/' + str(regul) + '/' + str(sizeofhidden))
                ploty.append(a.predict(testdata, testlabel))
    layers_loss(plotx,ploty)
    return max(list(acclist.values())), list(acclist.keys())[list(acclist.values()).index(max(list(acclist.values())))], acclist

def finalmodel(traindata, testdata, testlabel,lr, regu, hsize,batch_size):
    a = Sol(traindata, testdata, testlabel, lr, regu, hsize,batch_size)
    a.train(20)
    visualization_acc(a.trainaccuracy,a.testaccuracy,a.epoch+1)
    visualization_loss(a.loss_list,a.test_loss_list,a.epoch+1)
    a.savemodel()


    

if __name__ == '__main__':
    # 获取训练集/测试集 nparray型数据和标签
    traindata = images_data('train')
    trainlabel = labels_data('train')
    traindata = np.append(traindata, trainlabel, axis=1)

    testdata = images_data('test')
    testlabel = labels_data('test')


    output = findbest(traindata, testdata, testlabel,batch_size,learningRateList,L2_lambdas_list,sizeofhidden_list,5)  # 网格搜索找到较优的超参 
    best_model_parameters=output[1].split('/')
    print('the best model is ir ={} , regul={}, size of hidden={}'.format(best_model_parameters[0],best_model_parameters[1],best_model_parameters[2]))

    finalmodel(traindata, testdata, testlabel,float(best_model_parameters[0]),float(best_model_parameters[1]),int(best_model_parameters[2]),batch_size)
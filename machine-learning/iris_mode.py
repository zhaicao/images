import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


class LinearNet(nn.Module):
    """
    定义线性分类模型，继承父类nn.Module的所有属性和方法
    """
    def __init__(self, input_size, output_size):
        super(LinearNet, self).__init__()
        # 定义一层全连接层
        self.layer = nn.Linear(input_size, output_size)  # 四个输入三个输出
        # 定义softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer(x)
        out = self.softmax(out)
        return out


class Net(nn.Module):
    """
    定义两层神经网络
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.act = nn.Sigmoid()
        # self.act = nn.ReLU()  # 使用Relu激活函数
        # self.dropout = nn.Dropout(p=0.01)  # dropout训练

    def forward(self, x):
        out = self.layer1(x)
        # out = self.dropout(out)
        out = self.act(out)
        out = self.layer2(out)
        return out


class irisDataset(object):
    """
    数据集预处理类
    :param num_train: 训练集样本数
    :return:
    """

    def __init__(self, num_train, num_dev, num_test, batch_size=20):
        self.num_train = num_train
        self.num_dev = num_dev
        self.num_test = num_test
        self.batch_size = batch_size

    def loadDataset(self):
        dataSet = load_iris()
        # 归一化
        X = torch.from_numpy(dataSet['data']).float()
        bn = nn.BatchNorm1d(num_features=4, affine=False)
        # TensorDataset对tensor进行打包,该类中的tensor第一维度必须相等
        dataset = Data.TensorDataset(bn(X),
                                     torch.from_numpy(dataSet['target']).long())
        # 随机按比例划分数据集
        train_dataset, dev_dataset, test_dataset = Data.random_split(dataset,
                                                                     [self.num_train, self.num_dev, self.num_test])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)  # 将数据打乱

        dev_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                 shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  shuffle=True)
        return train_loader, test_loader, dev_loader

    def __call__(self, *args, **kwargs):
        return self.loadDataset()


class Evaluate(object):
    """
    评估类
    """
    def __init__(self, train_data, dev_data, test_data):
        super(Evaluate, self).__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

    def accuracy(self, model, data_loader):
        """
        预测评估准确度
        :param net: 网络
        :param data_loader: 验证集
        :return:
        """
        total = 0
        correct = 0
        for data, labels in data_loader:
            data = Variable(data)
            outputs = model(data)
            _, predicts = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicts == labels).sum()

        return (correct / total).item()

    # 绘制训练集和验证集的损失变化以及验证集上的准确率变化曲线
    def plot_training_loss_acc(self,
                               train_losses,
                               train_scores,
                               dev_losses,
                               dev_scores,
                               fig_name="fw-loss.pdf",
                               fig_size=(16, 6),
                               sample_step=20,
                               loss_legend_loc="upper right",
                               acc_legend_loc="lower right",
                               train_color="#8E004D",
                               dev_color='#E20079',
                               fontsize='x-large',
                               train_linestyle="-",
                               dev_linestyle='--'):

        plt.figure(figsize=fig_size)

        plt.subplot(1, 2, 1)
        train_epoch = [x[0] for x in train_losses]
        train_loss = [x[1] for x in train_losses]
        plt.plot(train_epoch, train_loss, color=train_color, linestyle=train_linestyle, label="Train loss")

        if len(dev_losses) > 0:
            dev_epoch = [x[0] for x in dev_losses]
            dev_loss = [x[1] for x in dev_losses]
            plt.plot(dev_epoch, dev_loss, color=dev_color, linestyle=dev_linestyle, label="Dev loss")
        # 绘制坐标轴和图例
        plt.ylabel("loss", fontsize=fontsize)
        plt.xlabel("epoch", fontsize=fontsize)
        plt.legend(loc=loss_legend_loc, fontsize=fontsize)

        # 绘制评价准确率变化曲线
        plt.subplot(1, 2, 2)
        train_epoch = [x[0] for x in train_scores]
        train_score = [x[1] for x in train_scores]
        plt.plot(train_epoch, train_score,
                 color=train_color, linestyle=train_linestyle, label="Train accuracy")

        if len(dev_scores) > 0:
            dev_epoch = [x[0] for x in dev_scores]
            dev_score = [x[1] for x in dev_scores]

            plt.plot(dev_epoch, dev_score,
                     color=dev_color, linestyle=dev_linestyle, label="Dev accuracy")

            # 绘制坐标轴和图例
            plt.ylabel("score", fontsize=fontsize)
            plt.xlabel("epoch", fontsize=fontsize)
            plt.legend(loc=acc_legend_loc, fontsize=fontsize)

        plt.savefig(fig_name)
        plt.show()

    # 计算损失和评价分数(准确度)
    def evaluate(self, model):
        # 用于统计训练/测试集的损失
        total_loss = 0
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        # 遍历验证集每个批次
        for batch_id, data in enumerate(self.dev_data):
            X, y = data
            # 计算模型输出
            logits = model(X)
            # 计算损失函数
            loss = criterion(logits, y).item()
            # 累积损失
            total_loss += loss

        dev_loss = (total_loss / len(self.dev_data))
        train_score = self.accuracy(model, self.train_data)
        dev_score = self.accuracy(model, self.dev_data)

        return train_score, dev_loss, dev_score

    def __call__(self, model):
        return self.evaluate(model)


class Runner(object):
    def __init__(self, model, dataSets, optimizer, loss_fn, **kwargs):
        self.model = model
        self.dataSets = dataSets
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # 记录训练过程中的评价指标变化情况
        self.dev_scores = []

        # 记录训练过程中的损失函数变化情况
        self.train_epoch_losses = []  # 一个epoch记录一次loss
        self.dev_epoch_losses = []  # 验证集loss记录

        self.train_epoch_scores = []
        self.dev_epoch_scores = []

    def train(self,
              model_paramters_file="results/model_parameter.pkl",
              fw_loss_file_name="results/fw-loss.pdf"):
        """
        训练网络
        :param model_paramters_file: 模型参数保存地址
        :param fw_loss_file_name:  损失及精度图片保存地址
        :return:
        """
        train_loader, dev_loader, test_loader = self.dataSets()
        evaluate = Evaluate(train_loader, dev_loader, test_loader)  # 定义评估类
        for epoch in range(num_epochs):
            print('current epoch = %d' % epoch)
            # 用于统计训练集的损失
            total_loss = 0
            for i, (data, labels) in enumerate(train_loader):  # 利用enumerate取出一个可迭代对象的内容
                data = Variable(data)
                labels = Variable(labels)
                outputs = net(data)  # 将数据集传入网络做前向计算
                loss = self.loss_fn(outputs, labels)  # 计算loss，不支持one hot编码，传入真实分类
                total_loss += loss  # 计算总loss
                self.optimizer.zero_grad()  # 在做反向传播之前先清除下网络状态
                loss.backward()  # loss反向传播
                self.optimizer.step()  # 更新参数

            # 当前epoch 训练loss累计
            trn_loss = (total_loss / len(train_loader)).item()
            # 当前epoch，验证集loss
            trn_score, drn_loss, dev_score = evaluate(net)
            print('current train loss = %.5f  dev loss = %.5f' % (trn_loss, drn_loss))

            # epoch粒度的训练loss保存
            self.train_epoch_losses.append((epoch, trn_loss))
            self.dev_epoch_losses.append((epoch, drn_loss))
            self.train_epoch_scores.append((epoch, trn_score))
            self.dev_epoch_scores.append((epoch, dev_score))

        torch.save(net.state_dict(), model_paramters_file)  # 保存模型参数
        print('finished training')

        # 测试集评估准确度
        model = Net(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(model_paramters_file))  # 读取模型参数
        accuracy = evaluate.accuracy(model, dev_loader)
        print('Test_dataset accuracy = %.2f' % (100 * accuracy) + '%')

        # 训练集和验证集损失和准确度可视化
        evaluate.plot_training_loss_acc(self.train_epoch_losses,
                                        self.train_epoch_scores,
                                        self.dev_epoch_losses,
                                        self.dev_epoch_scores,
                                        fw_loss_file_name)


if __name__ == '__main__':
    seed = 36  # 固定随机种子

    num_train = 100  # 训练集样本数
    num_dev = 25  # 验证集样本数
    num_test = 25  # 测试集样本数

    input_size = 4  # 输入层
    hidden_size = 6  # 隐藏层
    output_size = 3  # 输出层类别
    num_epochs = 500  # epoch
    batch_size = 20  # batchSize
    learning_rate = 0.1  # 学习率
    weight_decay = 0.001  # L2正则化
    model_paramters_file = "results/model_parameter_%s.pkl" % seed
    fw_loss_file_name = "results/fw-loss-%s.pdf" % seed

    torch.manual_seed(seed)

    dataSets = irisDataset(num_train=num_train, num_dev=num_dev, num_test=num_test)

    # 定义网络
    # net = LinearNet(input_size, output_size)  # Softmax线性网络
    net = Net(input_size, hidden_size, output_size)  # BP神经网络
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器，随机梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    runner = Runner(net, dataSets, optimizer, criterion)
    # 训练模型
    runner.train(model_paramters_file, fw_loss_file_name)

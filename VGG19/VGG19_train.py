import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import torchvision

import VGG19_preprocess
import VGG19_Net


# 超参数配置
config = {
    'train_batch': 32,  # 训练集大小
    'test_batch': 32,  # 测试集大小
    'net': VGG19_Net.VGG19,  # 网络判断类别需在VGG16_train.py处更改
    'learning_rate': 0.001,  # 学习率
    'epoch': 30,  # 训练轮次
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'loss_fn': nn.CrossEntropyLoss(),  # 交叉熵损失函数
}

train_dataloader = DataLoader(VGG19_preprocess.train_data, config['train_batch'], shuffle=True, num_workers=4)
test_dataloader = DataLoader(VGG19_preprocess.test_data, config['test_batch'], shuffle=False, num_workers=4)
# print('Train size is: {}\nTest size is: {}'.format(len(train_dataloader), len(test_dataloader)))

optimizer = optim.Adam(params=config['net'].parameters(), lr=config['learning_rate'])


def train_val(net, train, test, epochs, loss_fn, device):
    print("train on {}".format(device))
    best_acc = 0
    for epoch in range(epochs):
        current_time = time.strftime('%H:%M::%S', time.localtime(time.time()))
        print("%d--[%s] is training..." % (epoch + 1, current_time))

        # 模型训练
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAINING<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        net.train()
        net.to(device)
        total_train_loss = 0
        total_train_correct = 0
        train_num = 0

        for img, label, _ in tqdm(train):
            img = img.to(device)
            label = label.to(device)

            output = net(img)  # 预测结果

            loss = loss_fn(output, label)

            optimizer.zero_grad()  # 先梯度清理再进行优化
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            correct = (output.argmax(dim=1) == label).sum().item()  # 统计预测结合和真实结果相同的数量
            total_train_correct += correct
            train_num += len(img)

        train_acc = total_train_correct / train_num

        with open("log/train_loss.txt", 'a') as f:
            f.write("epoch {}--train:{}, loss:{}\n".format(epoch + 1, train_acc, total_train_loss))

        print("Train loss is:{}".format(total_train_loss))
        print("Train acc is:{}".format(train_acc))

        # 模型测试
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TESTING<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        net.eval()
        total_test_loss = 0
        total_test_correct = 0
        test_num = 0

        with torch.no_grad():  # 以下操作不影响网络参数
            for img, label, _ in tqdm(test):
                img = img.to(device)
                label = label.to(device)
                output = net(img)
                test_num += len(label)

                loss = loss_fn(output, label)
                total_test_loss = total_test_loss + loss.item()

                correct = (output.argmax(dim=1) == label).sum().item()  # 统计预测结合和真实结果相同的数量
                total_test_correct = total_test_correct + correct

            accuracy = total_test_correct / test_num

            print("Test loss is: {}".format(total_test_loss))
            print("Test acc is: {}".format(accuracy))

            if accuracy > best_acc:
                torch.save(net.state_dict(), 'weights/best_classify_vgg19.pt')

    torch.save(net.state_dict(), "weights/last_classify_vgg19.pt")


if __name__ == '__main__':
    train_val(config['net'], train_dataloader, test_dataloader, config['epoch'], config['loss_fn'], config['device'])

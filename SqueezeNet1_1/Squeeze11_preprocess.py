from abc import ABC  # todo：为什么要加这个？如何理解

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset as Dataset

train_txt_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\train.txt"
test_txt_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\test.txt"

train_high_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\train_high.txt"
test_high_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\test_high.txt"

# 图像预处理（训练和测试相同）
dataset_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)


# 读取数据集路径和label生成列表
def get_list(txt_path):
    path_list = []
    label_list = []
    with open(txt_path) as f:
        for i in f.readlines():
            path_list.append(i.split(',')[0])
            label_list.append(i.split(',')[1][:-1])  # split[1]切片后尾部带'\n'，再次切片切除最后一个字符可去除换行符
    return path_list, label_list


class SN11_low(Dataset, ABC):
    def __init__(self, trans, txt_path):
        self.transform = trans
        path_list, label_list = get_list(txt_path)
        self.path_list = path_list
        self.label_list = label_list

    def __getitem__(self, index):
        img = self.path_list[index]
        label = int(self.label_list[index])

        img = Image.open(img).convert('RGB')
        img = self.transform(img)

        label = torch.tensor(label)  # 图片正向传播计算出的预测值为tensor，与图片的标签作对比后进行反向传播，所以标签也要为tensor

        return img, label, index

    def __len__(self):
        return len(self.path_list)


train_data = SN11_low(dataset_transform, train_txt_path)
test_data = SN11_low(dataset_transform, test_txt_path)

high_train = SN11_low(dataset_transform, train_high_path)
high_test = SN11_low(dataset_transform, test_high_path)

if __name__ == '__main__':
    p, l = get_list(train_txt_path)
    print(l)

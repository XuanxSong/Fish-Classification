import torch
from tqdm import tqdm
import Squeeze11_Net
import Squeeze11_preprocess
from torch.utils.data import DataLoader
import time

# 更改此处参数测试不同的网络模型
best_weight_path = "./weights/best_classify_sn11.pt"
last_weight_path = "./weights/last_classify_vgg16.pt"
best_high_path = "./weights/best_classify_sn11_high.pt"
last_high_path = "./weights/last_classify_vgg16_high.pt"

test_data = Squeeze11_preprocess.test_data
high_test_data = Squeeze11_preprocess.high_test

data_loader = DataLoader(test_data, 1, shuffle=False, num_workers=4)

model = Squeeze11_Net.SqueezeNet1_1

device = "cpu"

# 加载模型开始测试
state_dict = torch.load(best_weight_path)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


def test(data):
    test_num = 0
    total_test_correct = 0
    with torch.no_grad():  # 以下操作不影响网络参数
        for img, label, _ in tqdm(data):
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            test_num += len(label)

            correct = (output.argmax(dim=1) == label).sum().item()  # 统计预测结合和真实结果相同的数量
            total_test_correct = total_test_correct + correct

        accuracy = total_test_correct / test_num

        print("Test train is: {}".format(accuracy))


if __name__ == '__main__':
    a = time.time()
    test(data_loader)
    b = time.time()
    print(b-a)
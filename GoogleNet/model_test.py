import torch
from tqdm import tqdm
import Google_Net
import Google_preprocess
import time
from torch.utils.data import DataLoader

# 更改此处参数测试不同的网络模型
best_weight_path = "./weights/best_classify.pt"
last_weight_path = "./weights/last_classify_vgg16.pt"
best_high_path = "./weights/best_classify_169_high.pt"
last_high_path = "./weights/last_classify_vgg16_high.pt"

test_data = Google_preprocess.test_data
high_test_data = Google_preprocess.high_test

data_loader = DataLoader(test_data, 1, shuffle=False, num_workers=4)

model = Google_Net.model

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
    start_time = time.time()
    test(data_loader)
    end_time = time.time()
    print(end_time-start_time)
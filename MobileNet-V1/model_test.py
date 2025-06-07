import torch
import tqdm
import V1_Net
import V1_preprocess

# 更改此处参数测试不同的网络模型
best_weight_path = "./weights/best_classify_v1.pt"
last_weight_path = "./weights/last_classify_v1.pt"
model = V1_Net.MobileNetV1()
test_data = V1_preprocess.test_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型开始测试
state_dict = torch.load(best_weight_path)
model.load_state_dict(state_dict)
model.eval()

test_num = 0
total_test_correct = 0

with torch.no_grad():  # 以下操作不影响网络参数
    for img, label, _ in tqdm(test_data):
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        test_num += len(label)

        correct = (output.argmax(dim=1) == label).sum().item()  # 统计预测结合和真实结果相同的数量
        total_test_correct = total_test_correct + correct

    accuracy = total_test_correct / test_num

    print("Test train is: {}".format(accuracy))

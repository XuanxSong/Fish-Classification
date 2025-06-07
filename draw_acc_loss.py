import matplotlib.pyplot as plt

resnet_low_path = r"C:\Users\31825\Desktop\Fish_Classify\ResNet18\log\train_loss.txt"
resnet_high_path = r"C:\Users\31825\Desktop\Fish_Classify\ResNet18\log\train_loss_high.txt"

googlenet_low_path = r"C:\Users\31825\Desktop\Fish_Classify\GoogleNet\log\train_loss.txt"
googlenet_high_path = r"C:\Users\31825\Desktop\Fish_Classify\GoogleNet\log\train_loss_high.txt"

densenet121_low_path = r"C:\Users\31825\Desktop\Fish_Classify\DenseNet\log\train_loss_121.txt"
densenet121_high_path = r"C:\Users\31825\Desktop\Fish_Classify\DenseNet\log\train_loss_121_high.txt"

mobilenetv2_low_path = r"C:\Users\31825\Desktop\Fish_Classify\MobileNet-V2\log\train_loss.txt"
mobilenetv2_high_path = r"C:\Users\31825\Desktop\Fish_Classify\MobileNet-V2\log\train_loss_high.txt"

shuffle05_low_path = r"C:\Users\31825\Desktop\Fish_Classify\ShuffleNet_V2\log\train_loss_x05.txt"
shuffle05_high_path = r"C:\Users\31825\Desktop\Fish_Classify\ShuffleNet_V2\log\train_loss_x05_high.txt"

shuffle10_low_path = r"C:\Users\31825\Desktop\Fish_Classify\ShuffleNet_V2\log\train_loss_x10.txt"
shuffle10_high_path = r"C:\Users\31825\Desktop\Fish_Classify\ShuffleNet_V2\log\train_loss_x10_high.txt"

squeeze11_low_path = r"C:\Users\31825\Desktop\Fish_Classify\SqueezeNet1_1\log\train_loss.txt"
squeeze11_high_path = r"C:\Users\31825\Desktop\Fish_Classify\SqueezeNet1_1\log\train_loss_high.txt"


def txt_2_list(txt_path):
    accuracy_list = []
    loss_list = []
    with open(txt_path, "r") as f:
        data_list = f.readlines()
        for data in data_list:
            acc = data.split(",")[0].split(":")[1]
            loss = data.split(",")[1].split(":")[1]
            # print(acc, loss)
            accuracy_list.append(round(float(acc), 3))
            loss_list.append(round(float(loss), 3))

    return accuracy_list, loss_list


# 绘制单幅网络， loss和acc分成两张图画
def draw_1(txt_path):
    acc_list, loss_list = txt_2_list(txt_path)
    epochs = range(len(loss_list))

    plt.figure(figsize=(12, 6))  # 设置画布尺寸

    # 绘制loss折线图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_list, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制accuracy折线图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_list, 'r', label='Training accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 绘制单个网络，loss和acc一起
def draw_2(low_path, high_path):
    acc_low_list, loss_low_list = txt_2_list(low_path)
    acc_high_list, loss_high_list = txt_2_list(high_path)
    epochs = range(1, len(acc_high_list)+1)

    # 设置图像整体大小
    fig, ax1 = plt.subplots(figsize=(24, 18))

    # 绘制loss曲线
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=30)
    ax1.set_ylabel('Loss', color='black', fontsize=30)
    ax1.plot(epochs, loss_low_list, color=color, linewidth=3)
    ax1.tick_params(axis='y', labelcolor='black')
    color = 'tab:red'
    ax1.plot(epochs, loss_high_list, color=color, linewidth=3)

    plt.tick_params(axis='both', labelsize=20)

    # 创建第二个Y轴并绘制accuracy曲线
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color='black', fontsize=30)
    # ax2.set_ylim(0.5)
    ax2.plot(epochs, acc_low_list, color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor='black')
    color = 'tab:red'
    ax2.plot(epochs, acc_high_list, color=color, linewidth=3)

    # 添加图例
    plt.legend(['Low Resolution', 'Super Resolution'], loc=7, fontsize=25)
    # 更改坐标轴字体大小
    plt.tick_params(axis='both', labelsize=20)
    plt.show()


def draw_3(net_1, net_2, net_3, net_4, net_5, net_6):
    acc_1, loss_1 = txt_2_list(net_1)
    acc_2, loss_2 = txt_2_list(net_2)
    acc_3, loss_3 = txt_2_list(net_3)
    acc_4, loss_4 = txt_2_list(net_4)
    acc_5, loss_5 = txt_2_list(net_5)
    acc_6, loss_6 = txt_2_list(net_6)
    epochs = range(1, len(acc_1)+1)

    plt.figure(figsize=(24, 18))
    plt.plot(epochs, loss_1, label='ResNet18', color='red', linewidth=3)
    plt.plot(epochs, loss_2, label='GoogleNet', color='blue', linewidth=3)
    plt.plot(epochs, loss_3, label='DenseNet121', color='green', linewidth=3)
    plt.plot(epochs, loss_4, label='MobileNetV2', color='purple', linewidth=3)
    plt.plot(epochs, loss_5, label='ShuffleNetV2_x0.5', color='orange', linewidth=3)
    plt.plot(epochs, loss_6, label='SqueezeNet1_1', color='black', linewidth=3)

    plt.legend(fontsize=25)

    plt.xlabel('Epochs', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.tick_params(axis='both', labelsize=20)

    plt.show()


def draw_4(net_1, net_1h, net_2, net_2h, net_3, net_3h, net_4, net_4h, net_5, net_5h, net_6, net_6h):
    acc_1, loss_1 = txt_2_list(net_1)
    acc_1h, loss_1h = txt_2_list(net_1h)
    acc_2, loss_2 = txt_2_list(net_2)
    acc_2h, loss_2h = txt_2_list(net_2h)
    acc_3, loss_3 = txt_2_list(net_3)
    acc_3h, loss_3h = txt_2_list(net_3h)
    acc_4, loss_4 = txt_2_list(net_4)
    acc_4h, loss_4h = txt_2_list(net_4h)
    acc_5, loss_5 = txt_2_list(net_5)
    acc_5h, loss_5h = txt_2_list(net_5h)
    acc_6, loss_6 = txt_2_list(net_6)
    acc_6h, loss_6h = txt_2_list(net_6h)
    epochs = range(len(acc_1))

    plt.figure(figsize=(15, 6))
    plt.plot(epochs, loss_1, label='ResNet18', color='red', linestyle='-.')
    plt.plot(epochs, loss_1h, label='ResNet18(HR)', color='red', linestyle='-')
    plt.plot(epochs, loss_2, label='DenseNet121', color='blue', linestyle='-.')
    plt.plot(epochs, loss_2h, label='DenseNet121(HR)', color='blue', linestyle='-')
    plt.plot(epochs, loss_3, label='GoogleNet', color='green', linestyle='-.')
    plt.plot(epochs, loss_3h, label='GoogleNet(HR)', color='green', linestyle='-')
    plt.plot(epochs, loss_4, label='MobileNetV2', color='purple', linestyle='-.')
    plt.plot(epochs, loss_4h, label='MobileNetV2(HR)', color='purple', linestyle='-')
    plt.plot(epochs, loss_5, label='ShuffleNetV2_x0.5', color='orange', linestyle='-.')
    plt.plot(epochs, loss_5h, label='ShuffleNetV2_x0.5(HR)', color='orange', linestyle='-')
    plt.plot(epochs, loss_6, label='SqueezeNet1_1', color='black', linestyle='-.')
    plt.plot(epochs, loss_6h, label='SqueezeNet1_1(HR)', color='black', linestyle='-')

    plt.legend()

    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('HR VS LR(Loss)', fontsize=20)

    plt.show()


if __name__ == '__main__':
    # draw_4(resnet_low_path, resnet_high_path, densenet121_low_path, densenet121_high_path, googlenet_low_path,
    # googlenet_high_path, mobilenetv2_low_path, mobilenetv2_high_path, shuffle05_low_path, shuffle05_high_path,
    # shuffle10_low_path, shuffle10_high_path)
    # draw_2(squeeze11_low_path, squeeze11_high_path)
    # draw_2(resnet_low_path, resnet_high_path)
    draw_3(resnet_low_path, googlenet_low_path, densenet121_low_path, mobilenetv2_low_path, shuffle05_low_path, squeeze11_low_path)

import warnings
import torch.nn as nn
import torchvision

warnings.simplefilter("ignore", UserWarning)
MobileNetV2 = torchvision.models.mobilenet_v2(pretrained=True)
warnings.resetwarnings()


print(MobileNetV2)

MobileNetV2.classifier = nn.Sequential(
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 23)
)



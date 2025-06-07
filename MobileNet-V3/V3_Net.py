import warnings
import torchvision
import torch.nn as nn

warnings.simplefilter("ignore", UserWarning)
MobileNetV3_small = torchvision.models.mobilenet_v3_small(pretrained=True)
MobileNetV3_large = torchvision.models.mobilenet_v3_large(pretrained=True)
warnings.resetwarnings()

num_ftrs_small = MobileNetV3_small.classifier[-1].in_features
MobileNetV3_small.classifier[-1] = nn.Linear(num_ftrs_small, 23)

num_ftrs_large = MobileNetV3_large.classifier[-1].in_features
MobileNetV3_large.classifier[-1] = nn.Linear(num_ftrs_large, 23)

# print(MobileNetV3_small)
# print(MobileNetV3_large)
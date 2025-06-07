import torchvision
import warnings
import torch.nn as nn


warnings.simplefilter("ignore", UserWarning)
ShuffleNet_x10 = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
warnings.resetwarnings()

ShuffleNet_x10.fc = nn.Linear(ShuffleNet_x10.fc.in_features, 23)
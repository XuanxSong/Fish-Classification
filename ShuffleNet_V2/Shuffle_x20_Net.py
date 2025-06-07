import torchvision
import warnings
import torch.nn as nn


warnings.simplefilter("ignore", UserWarning)
ShuffleNet_x20 = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
warnings.resetwarnings()

ShuffleNet_x20.fc = nn.Linear(ShuffleNet_x20.fc.in_features, 23)
import torchvision
import warnings
import torch.nn as nn


warnings.simplefilter("ignore", UserWarning)
ShuffleNet_x15 = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
warnings.resetwarnings()

ShuffleNet_x15.fc = nn.Linear(ShuffleNet_x15.fc.in_features, 23)
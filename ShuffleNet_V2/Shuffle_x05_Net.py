import torchvision
import warnings
import torch.nn as nn

warnings.simplefilter("ignore", UserWarning)
ShuffleNet_x05 = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
warnings.resetwarnings()

ShuffleNet_x05.fc = nn.Linear(ShuffleNet_x05.fc.in_features, 23)

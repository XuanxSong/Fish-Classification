import torchvision
import warnings
import torch.nn as nn


warnings.simplefilter("ignore", UserWarning)
SqueezeNet1_0 = torchvision.models.squeezenet1_0(pretrained=True)
warnings.resetwarnings()

SqueezeNet1_0.classifier[1] = nn.Conv2d(512, 23, kernel_size=(1, 1))
SqueezeNet1_0.num_classes = 23
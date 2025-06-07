import torchvision
import warnings
import torch.nn as nn

warnings.simplefilter("ignore", UserWarning)
SqueezeNet1_1 = torchvision.models.squeezenet1_1(pretrained=True)
warnings.resetwarnings()

SqueezeNet1_1.classifier[1] = nn.Conv2d(512, 23, kernel_size=(1, 1))
SqueezeNet1_1.num_classes = 23

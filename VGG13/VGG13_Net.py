import warnings
import torch.nn as nn
import torchvision


warnings.simplefilter("ignore", UserWarning)
VGG13 = torchvision.models.vgg13(pretrained=True)
warnings.resetwarnings()

num_classes = 23
VGG13.classifier[6] = nn.Linear(4096, num_classes)
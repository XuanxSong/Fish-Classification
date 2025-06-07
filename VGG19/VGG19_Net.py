import warnings
import torch.nn as nn
import torchvision


warnings.simplefilter("ignore", UserWarning)
VGG19 = torchvision.models.vgg19(pretrained=True)
warnings.resetwarnings()

num_classes = 23
VGG19.classifier[6] = nn.Linear(4096, num_classes)

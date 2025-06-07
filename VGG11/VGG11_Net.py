import warnings
import torch.nn as nn
import torchvision

warnings.simplefilter("ignore", UserWarning)
VGG11 = torchvision.models.vgg11(pretrained=True)
warnings.resetwarnings()

num_classes = 23
VGG11.classifier[6] = nn.Linear(4096, num_classes)

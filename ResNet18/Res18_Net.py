import warnings
import torch.nn as nn
import torchvision

warnings.simplefilter("ignore", UserWarning)
model = torchvision.models.resnet18(pretrained=True)
warnings.resetwarnings()


print(model)
num_classes = 23
model.fc = nn.Linear(model.fc.in_features, num_classes)
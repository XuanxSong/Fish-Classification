import warnings
import torchvision
import torch.nn as nn


warnings.simplefilter("ignore", UserWarning)
model = torchvision.models.resnet34(pretrained=True)
warnings.resetwarnings()

model.fc = nn.Linear(model.fc.in_features, 23)
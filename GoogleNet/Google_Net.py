import torchvision
import warnings
import torch.nn as nn


warnings.simplefilter("ignore", UserWarning)
model = torchvision.models.googlenet(pretrained=True)
warnings.resetwarnings()

model.fc = nn.Linear(model.fc.in_features, 23)
# print(model)
import torchvision
import warnings
import torch.nn as nn

warnings.simplefilter("ignore", UserWarning)
model = torchvision.models.densenet121(pretrained=True)
warnings.resetwarnings()

model.classifier = nn.Linear(model.classifier.in_features, 23)
# print(model)

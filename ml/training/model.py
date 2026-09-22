from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


from torchvision.models import resnet18
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class Classifier(nn.Module):
    def __init__(self, latent_channels):
        super(Classifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(50 * 5 * 5, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, z, alpha):
        z = ReverseLayerF.apply(z, alpha)
        z = self.feature_extractor(z)
        z = z.view(z.size(0), -1)  # Flatten
        output = self.classifier(z)
        return output

class SimpleClassifier(nn.Module):
    def __init__(self, latent_channels):
        super(SimpleClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(50 * 5 * 5, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, z):
        z = self.feature_extractor(z)
        z = z.view(z.size(0), -1)  # Flatten
        output = self.classifier(z)
        return output

class SimpleMLP(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# class ResnetClassifier(nn.Module):
#     def __init__(self, latent_channels, num_classes):
#         super(ResnetClassifier, self).__init__()
#         self.model = resnet18(pretrained=True)
#         # Replace the first convolution layer to match the latent shape (3x32x32)
#         self.model.conv1 = nn.Conv2d(latent_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         # Replace the output layer for binary classification
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)


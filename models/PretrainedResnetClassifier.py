from torchvision.models import resnet18
import torch.nn as nn

# class Classifier(nn.Module):
#     def __init__(self, latent_channels, num_classes):
#         super(Classifier, self).__init__()
#         self.model = resnet18(pretrained=True)
#         # Replace the first convolution layer to match the latent shape (3x32x32)
#         self.model.conv1 = nn.Conv2d(latent_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         # Replace the output layer for binary classification
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)


class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


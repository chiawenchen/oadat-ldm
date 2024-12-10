
from lightning.pytorch import LightningModule
import torch.nn as nn
from torchvision import models
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
import torch
from torch.optim.lr_scheduler import LinearLR

class ResNetClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4, num_timesteps=1000, embedding_dim=128):
        super(ResNetClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim

        # Load pre-trained ResNet and adapt for grayscale input and binary classification
        self.model = models.resnet18(weights='DEFAULT')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for 1-channel input

        # Remove the original fully connected layer (fc) and replace it with our custom layer
        self.feature_dim = self.model.fc.in_features  # Typically 512
        self.model.fc = nn.Identity()  # Replace fc layer with identity to prevent its application in forward pass

        # Custom classification layer to accommodate concatenated image and timestep features
        self.fc = nn.Linear(self.feature_dim + embedding_dim, num_classes)

        # Timestep embedding layer
        self.timestep_embedding = nn.Sequential(
            nn.Embedding(num_timesteps, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()

    def forward(self, x, timesteps):
        # Extract features using ResNet up to the replaced fc layer
        img_features = self.model(x)  # Now outputs shape [batch_size, 512]

        # Embed the timestep and reshape for concatenation
        timestep_embed = self.timestep_embedding(timesteps)  # Shape: [batch_size, 128]

        # Concatenate image features with timestep embeddings along the last dimension
        combined_features = torch.cat([img_features, timestep_embed], dim=1)  # Shape: [batch_size, 640]

        # Pass the combined features through the final classification layer
        return self.fc(combined_features)

    def training_step(self, batch, batch_idx):
        images, labels, timesteps = batch
        outputs = self(images, timesteps)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Compute metrics
        self.train_accuracy(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.train_f1(preds, labels)

        # Log metrics to progress bar
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, prog_bar=False)
        self.log("train_precision", self.train_precision, prog_bar=True)
        self.log("train_recall", self.train_recall, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, timesteps = batch
        outputs = self(images, timesteps)
        preds = torch.argmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        # Compute metrics
        self.val_accuracy(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)

        # Log metrics to progress bar
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, prog_bar=False)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

        return loss
    
    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = LinearLR(
            optimizer, total_iters=1, last_epoch=-1
        )
        return [optimizer], [lr_scheduler]
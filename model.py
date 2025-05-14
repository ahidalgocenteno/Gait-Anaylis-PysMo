import torch
import torch.nn as nn
import torch.nn.functional as F

class GaitModel(nn.Module):
    def __init__(self, num_classes, input_size=(96, 64, 44)):
        super(GaitModel, self).__init__()
        
        # 3D CNN Backbone
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        # compute size after convs
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_size)
            out = self.backbone(dummy)
            _, C, T, H, W = out.shape

        # Temporal Tokenization
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=C * H * W, nhead=4, dim_feedforward=512, dropout=0.1),
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(C * H * W, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, T, H, W)
        features = self.backbone(x)  # -> (B, C, T', H', W')
        B, C, T, H, W = features.shape

        # flatten spatial dims
        features = features.permute(2, 0, 1, 3, 4).contiguous()  # (T, B, C, H, W)
        features = features.view(T, B, C * H * W)  # (T, B, D)

        # transformer encoding
        encoded = self.temporal_encoder(features)  # (T, B, D)

        # Use mean pooling over time
        pooled = encoded.mean(dim=0)  # (B, D)

        return self.classifier(pooled)  # (B, num_classes)

        
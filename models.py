import torch
import torch.nn as nn


class MeanTeacherModel(nn.Module):
    """
    Combines a backbone encoder and a classification head.
    """

    def __init__(self, encoder, num_classes, dropout_rate=0.5):
        super().__init__()
        self.encoder = encoder

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = encoder(dummy_input)
            encode_output_features = dummy_output.view(dummy_output.size(0), -1).size(1)


        self.dropout = nn.Dropout()
        self.classification_head = nn.Linear(encode_output_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        logits = self.classification_head(features)
        return logits


class FixMatchModel(nn.Module):
    """
    FixMatch Model.
    Combines a backbone encoder and a classification head.
    """

    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = encoder(dummy_input)
            encode_output_features = dummy_output.view(dummy_output.size(0), -1).size(1)

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(encode_output_features, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        logits = self.classification_head(features)
        return logits
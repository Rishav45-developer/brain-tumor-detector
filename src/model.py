

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Feature extraction: 3 convolutional blocks, each shrinking
        # the image size while increasing the number of learned features
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
        )

        # Classification head: takes the extracted features and
        # outputs a score for each of the 4 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import torch

    model = SimpleCNN(num_classes=4)
    print(model)

    
    fake_batch = torch.randn(8, 3, 224, 224)  
    output = model(fake_batch)
    print(f"\nOutput shape: {output.shape}")  
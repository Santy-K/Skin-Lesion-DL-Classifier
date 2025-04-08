import torch.nn as nn
class ConvNetModel_1(nn.Module):
    """Basic Convolutional Neural Network model for image classification based on hw1 of COMP9444.
    """
    def __init__(self, num_classes = 3):
        """Basic Convolutional Neural Network model for image classification based on hw1 of COMP9444.
        This model is a simple CNN with two convolutional layers followed by a fully connected layer.
        

        Args:
            num_classes (int, optional): Number of classes to use. Defaults to 3.
        """
        super(ConvNetModel_1, self).__init__()
        self.layer1 = nn.Sequential(
            #Because is RGB in channells = 3 
            nn.Conv2d(3, 12, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # For 128x128 -> /2 -> 64 -> /2 -> 32 => final feature map: (32 filters, 32 x 32).
        self.fc = nn.Linear(32*32*32, num_classes)
        
    def forward(self, x):
        """Forward pass of the model."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
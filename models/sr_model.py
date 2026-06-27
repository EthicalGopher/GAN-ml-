import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN)
    Based on Dong et al. (https://arxiv.org/abs/1409.1556)
    
    This model takes a low-resolution image (upscaled via bicubic interpolation)
    and learns a mapping to output a high-resolution reconstructed image.
    """
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # 1. Patch extraction and representation: Conv(9x9, 64 filters)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 2. Non-linear mapping: Conv(5x5, 32 filters)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 3. Reconstruction: Conv(5x5, num_channels filters)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out

import torch
import torch.nn as nn
import torchvision


class block(nn.Module): 
    def __init__(self, channels, projection_shortcut = None):
        super(block, self).__init__()

        # padding = (kernel_size-1) / 2
        if projection_shortcut is None:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1) # diff: strdie = 2
            
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
        self.projection_shortcut = projection_shortcut

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.projection_shortcut is not None:
            identity = self.projection_shortcut(identity)
            pass


        x += identity
        x = self.relu(x)
        return x


# ResNet34_B: 
# projection shortcut only for increasing dimansion, 
# other shortcut is identity
# (and ResNet50/101 also based on B)
class ResNet34(nn.Module):
    def __init__(self, in_channels):
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride = 2)

        self.avgpool = nn.AvgPool2d(kernel_size=3)

    # num_of_blocks_per_layers = [3,4,6,3]
    def create_layer(self, block, channels, num_of_blocks, downsample = False):
        layers = [] # archtecture of net
        
        projection_shortcut = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(channels)
        )

        if downsample is False:
            layers.append(block(channels))
        else:
            layers.append(block(channels, projection_shortcut))    
        for i in range(num_of_blocks-1):
            layers.append(block(channels))
                
                
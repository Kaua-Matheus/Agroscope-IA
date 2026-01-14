# Torch
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


## Corn
def CornSetter():

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            # Construção das hidden layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # 128
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # 30
            self.bn3 = nn.BatchNorm2d(128)

            self.pool = nn.MaxPool2d(2, 2) # 15

            # Flatten
            self.fc1 = nn.Linear(128 * 15 * 15, 120)
            self.bn4 = nn.BatchNorm1d(120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 4)

            # Ativação GeLU
            self.gelu = nn.GELU()
        
        def forward(self, x):
            x = self.pool(F.gelu(self.bn1(self.conv1(x))))
            x = self.pool(F.gelu(self.bn2(self.conv2(x))))
            x = self.pool(F.gelu(self.bn3(self.conv3(x))))

            x = torch.flatten(x, 1)

            x = F.gelu(self.bn4(self.fc1(x)))
            x = F.gelu(self.fc2(x))
            x = self.fc3(x)

            return x


    ## Transform ##

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ## Retorno das Instâncias
    return NeuralNetwork, transform


## Soybean
def SoybeanSetter():

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            # Construção das hidden layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3) 
            self.bn3 = nn.BatchNorm2d(128)

            self.pool = nn.MaxPool2d(2, 2)

            # Flatten
            self.fc1 = nn.Linear(128 * 15 * 15, 120)
            self.bn4 = nn.BatchNorm1d(120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

            # Ativação GeLU
            self.gelu = nn.GELU()
        
        def forward(self, x):
            x = self.pool(F.gelu(self.bn1(self.conv1(x))))
            x = self.pool(F.gelu(self.bn2(self.conv2(x))))
            x = self.pool(F.gelu(self.bn3(self.conv3(x))))

            x = torch.flatten(x, 1)

            x = F.gelu(self.bn4(self.fc1(x)))
            x = F.gelu(self.fc2(x))
            x = self.fc3(x)

            return x


    ## Transform ##

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ## Retorno das Instâncias
    return NeuralNetwork, transform


## Wheat
def WheatSetter():

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            # Construção das hidden layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
            self.bn3 = nn.BatchNorm2d(128)

            self.pool = nn.MaxPool2d(2, 2) # 15

            # Flatten
            self.fc1 = nn.Linear(128 * 15 * 15, 120)
            self.bn4 = nn.BatchNorm1d(120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 7)

            # Ativação GeLU
            self.gelu = nn.GELU()
        
        def forward(self, x):
            x = self.pool(F.gelu(self.bn1(self.conv1(x))))
            x = self.pool(F.gelu(self.bn2(self.conv2(x))))
            x = self.pool(F.gelu(self.bn3(self.conv3(x))))

            x = torch.flatten(x, 1)

            x = F.gelu(self.bn4(self.fc1(x)))
            x = F.gelu(self.fc2(x))
            x = self.fc3(x)

            return x
        

    ## Transform ##
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    ## Retorno das Instâncias
    return NeuralNetwork, transform
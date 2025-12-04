import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43): # GTSRB'de 43 sınıf var
        super(TrafficSignNet, self).__init__()
        
        # 1. Evrişim Bloğu
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0), # Giriş: 3 kanal (RGB)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2. Evrişim Bloğu
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Tam Bağlantılı Katmanlar (Fully Connected)
        # 32x32 giriş -> layer1 sonrası 14x14 -> layer2 sonrası 5x5
        self.fc1 = nn.Linear(64 * 5 * 5, 120) 
        self.dropout = nn.Dropout(0.5) # Overfitting'i önlemek için
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # Flatten (Düzleştirme)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

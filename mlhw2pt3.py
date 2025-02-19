import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Check if CUDA is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Residual Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the ResNet-18 Model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Define transforms for CIFAR-10/100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 and CIFAR-100 datasets
trainset_10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset_10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainset_100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset_100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader_10 = DataLoader(trainset_10, batch_size=64, shuffle=True)
testloader_10 = DataLoader(testset_10, batch_size=64, shuffle=False)

trainloader_100 = DataLoader(trainset_100, batch_size=64, shuffle=True)
testloader_100 = DataLoader(testset_100, batch_size=64, shuffle=False)

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_loss, val_loss, val_acc = [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Train the model
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        
        # Validation accuracy
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss_epoch = val_running_loss / len(test_loader)
        val_loss.append(val_loss_epoch)
        
        val_acc_epoch = val_correct / val_total
        val_acc.append(val_acc_epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}')
    
    return train_loss, val_loss, val_acc

# Initialize model, criterion, and optimizer
model_10 = ResNet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_10.parameters(), lr=0.001)

# Train on CIFAR-10
#train_loss_10, val_loss_10, val_acc_10 = train_model(model_10, trainloader_10, testloader_10, criterion, optimizer, num_epochs=10)

# Plot Training Results for CIFAR-10
plt.figure(figsize=(10, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_loss_10, label='Train Loss')
plt.plot(val_loss_10, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(val_acc_10, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Train on CIFAR-100 (you can use the same ResNet18 model or create a new one)
model_100 = ResNet18(num_classes=100).to(device)  # Update the number of classes for CIFAR-100
optimizer_100 = optim.Adam(model_100.parameters(), lr=0.001)

# Train the model on CIFAR-100
#train_loss_100, val_loss_100, val_acc_100 = train_model(model_100, trainloader_100, testloader_100, criterion, optimizer_100, num_epochs=10)

# Plot Training Results for CIFAR-100
plt.figure(figsize=(10, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_loss_100, label='Train Loss')
plt.plot(val_loss_100, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(val_acc_100, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

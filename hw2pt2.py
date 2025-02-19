import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for CIFAR-10 and CIFAR-100
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
trainset_c10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset_c10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainset_c100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset_c100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# DataLoaders
trainloader_c10 = DataLoader(trainset_c10, batch_size=64, shuffle=True, num_workers=2)
testloader_c10 = DataLoader(testset_c10, batch_size=64, shuffle=False, num_workers=2)

trainloader_c100 = DataLoader(trainset_c100, batch_size=64, shuffle=True, num_workers=2)
testloader_c100 = DataLoader(testset_c100, batch_size=64, shuffle=False, num_workers=2)

# Define the VGG model
class CustomVGG(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CustomVGG, self).__init__()
        self.vgg = vgg16(pretrained=False)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.vgg(x)

# Training function
def train(model, trainloader, testloader, epochs=15, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}: Train Loss: {running_loss/len(trainloader):.4f}, "
              f"Val Loss: {val_loss/len(testloader):.4f}, Val Acc: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    model_c10 = CustomVGG(num_classes=10).to(device)
    print("Training VGG on CIFAR-10")
    train(model_c10, trainloader_c10, testloader_c10)

    model_c100 = CustomVGG(num_classes=100).to(device)
    print("Training VGG on CIFAR-100")
    train(model_c100, trainloader_c100, testloader_c100)

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Define a simplified AlexNet for CIFAR-10/100
    class AlexNetCIFAR(nn.Module):
        def __init__(self, num_classes=10, use_dropout=False):
            super(AlexNetCIFAR, self).__init__()
            self.use_dropout = use_dropout
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),

                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),

                nn.Linear(512, num_classes)  # Adjust for CIFAR-10 or CIFAR-100
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Model parameter count function
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define transforms (Same for both CIFAR-10 and CIFAR-100)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    trainset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader_cifar10 = DataLoader(trainset_cifar10, batch_size=128, shuffle=True, num_workers=0)
    testloader_cifar10 = DataLoader(testset_cifar10, batch_size=100, shuffle=False, num_workers=0)

    # Load CIFAR-100 dataset
    trainset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    trainloader_cifar100 = DataLoader(trainset_cifar100, batch_size=128, shuffle=True, num_workers=0)
    testloader_cifar100 = DataLoader(testset_cifar100, batch_size=100, shuffle=False, num_workers=0)

    # Training function with Learning Rate Scheduler
    def train_model(model, trainloader, testloader, epochs=30, lr=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            correct, total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {running_loss/len(trainloader):.4f}, "
                  f"Val Loss: {val_loss/len(testloader):.4f}, Val Acc: {100 * correct / total:.2f}%")

            scheduler.step()  # Update learning rate

    # Train CIFAR-10 model
    print("\nTraining CIFAR-10 Model (Without Dropout)...")
    model_cifar10_no_dropout = AlexNetCIFAR(num_classes=10, use_dropout=False)
    print(f"Params in modified AlexNet for CIFAR-10: {count_parameters(model_cifar10_no_dropout)}")
    train_model(model_cifar10_no_dropout, trainloader_cifar10, testloader_cifar10)

    print("\nTraining CIFAR-10 Model (With Dropout)...")
    model_cifar10_dropout = AlexNetCIFAR(num_classes=10, use_dropout=True)
    train_model(model_cifar10_dropout, trainloader_cifar10, testloader_cifar10)

    # Train CIFAR-100 model
    print("\nTraining CIFAR-100 Model (Without Dropout)...")
    model_cifar100_no_dropout = AlexNetCIFAR(num_classes=100, use_dropout=False)
    print(f"Params in modified AlexNet for CIFAR-100: {count_parameters(model_cifar100_no_dropout)}")
    train_model(model_cifar100_no_dropout, trainloader_cifar100, testloader_cifar100)

    print("\nTraining CIFAR-100 Model (With Dropout)...")
    model_cifar100_dropout = AlexNetCIFAR(num_classes=100, use_dropout=True)
    train_model(model_cifar100_dropout, trainloader_cifar100, testloader_cifar100)

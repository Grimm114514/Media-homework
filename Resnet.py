import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from collections import OrderedDict 
import pandas as pd  
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("Using device:", device)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet_type='resnet50', pretrained=True, use_avgpool=False):
        super(ResNetFeatureExtractor, self).__init__() 
        
        # Load a pre-trained ResNet model
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnet_type == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnet_type == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Unsupported ResNet type: {}".format(resnet_type))
        
        # Remove the fully connected layer
        self.features = nn.Sequential(OrderedDict([
            ('conv1', self.resnet.conv1),
            ('bn1', self.resnet.bn1),
            ('relu', self.resnet.relu),
            ('maxpool', self.resnet.maxpool),
            ('layer1', self.resnet.layer1),
            ('layer2', self.resnet.layer2),
            ('layer3', self.resnet.layer3),
            ('layer4', self.resnet.layer4),
        ]))
        
        self.use_avgpool = use_avgpool
        if use_avgpool:
            self.avgpool = self.resnet.avgpool

    def forward(self, x):
        x = self.features(x)
        if self.use_avgpool:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# 定义 ResNet 模型
class CIFAR10ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10ResNet, self).__init__()
        self.resnet = resnet50(pretrained=False)  # 使用 ResNet-50
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # 修改最后一层以适配 CIFAR-10

    def forward(self, x):
        return self.resnet(x)

model = CIFAR10ResNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)# 优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)# 学习率调度器

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# 测试函数
def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

if __name__ == '__main__':
    # 数据加载和训练代码
    num_epochs = 30
    history = []  # 用于记录每个 epoch 的训练和测试结果
    if os.path.exists('checkpoint_1.pth'):
        model.load_state_dict(torch.load('checkpoint_1.pth'))

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        scheduler.step()

        # 记录训练和测试结果
        history.append({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Test Loss': test_loss,
            'Test Accuracy': test_acc
        })

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 使用 matplotlib 绘制训练和测试的损失、准确率曲线
    epochs = [h['Epoch'] for h in history]
    train_losses = [h['Train Loss'] for h in history]
    test_losses = [h['Test Loss'] for h in history]
    train_accuracies = [h['Train Accuracy'] for h in history]
    test_accuracies = [h['Test Accuracy'] for h in history]

    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()

    # 保存图表到文件
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    # 保存最终模型
    torch.save(model.state_dict(), f'checkpoint_1.pth')

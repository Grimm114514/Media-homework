import math  
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import time


#ACTIVATION_CHOICE = "ReLU"
ACTIVATION_CHOICE = "LeakyReLU"
#ACTIVATION_CHOICE="GELU"
#ACTIVATION_CHOICE="Sigmoid"

# --- B. 选择优化器 (取消注释你想要的那一行) ---
OPTIMIZER_CHOICE = "SGD"             # 纯 SGD
#OPTIMIZER_CHOICE = "SGD_Momentum"    # SGD + 0.9 Momentum
#OPTIMIZER_CHOICE = "Adam"            # Adam


BATCH_SIZE = 128
LEARNING_RATE = 0.01  
EPOCHS = 40          
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"当前实验配置: 激活函数 [{ACTIVATION_CHOICE}] | 优化器 [{OPTIMIZER_CHOICE}] | 设备 [{DEVICE}]")

def get_data_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        
        init_channels = math.ceil(oup / ratio)
        
        new_channels = init_channels * (ratio - 1)

        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

       
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :] 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation_class=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.conv1 = GhostModule(in_planes, planes, kernel_size=3, stride=stride, ratio=2)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = activation_class()
        
        self.conv2 = GhostModule(planes, planes, kernel_size=3, stride=1, ratio=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = activation_class()
        # ------------------------------------------------

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
               
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation_class=nn.ReLU):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.activation_class = activation_class
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = activation_class()
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.activation_class))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(activation_class):
    return ResNet(BasicBlock, [2, 2, 2, 2], activation_class=activation_class)

if ACTIVATION_CHOICE == "ReLU":
    act_class = nn.ReLU
elif ACTIVATION_CHOICE == "LeakyReLU":
    act_class = nn.LeakyReLU  
elif ACTIVATION_CHOICE == "GELU":
    act_class = nn.GELU
elif ACTIVATION_CHOICE == "Sigmoid":
    act_class = nn.Sigmoid
else:
    raise ValueError("未知的激活函数选择")


model = ResNet18(activation_class=act_class).to(DEVICE)


criterion = nn.CrossEntropyLoss()


if OPTIMIZER_CHOICE == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0, weight_decay=5e-4)
elif OPTIMIZER_CHOICE == "SGD_Momentum":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
elif OPTIMIZER_CHOICE == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
else:
    raise ValueError("未知的优化器选择")

def train():
    trainloader, testloader = get_data_loaders()
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    print("开始训练...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_train_acc = 100. * correct / total
        loss_history.append(epoch_loss)
        train_acc_history.append(epoch_train_acc)
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        epoch_test_acc = 100. * test_correct / test_total
        test_acc_history.append(epoch_test_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Test Acc: {epoch_test_acc:.2f}%")

    elapsed_time = time.time() - start_time
    print(f"训练完成，耗时: {elapsed_time:.1f}s")
    return loss_history, train_acc_history, test_acc_history, elapsed_time

def plot_and_save(loss_history, train_acc_history, test_acc_history, elapsed_time):
    # 确保文件夹存在
    save_dir = "figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(range(1, EPOCHS + 1), loss_history, marker='o', label=f'{ACTIVATION_CHOICE} + {OPTIMIZER_CHOICE}')
    axes[0].set_title(f'Training Loss Curve ({ACTIVATION_CHOICE} + {OPTIMIZER_CHOICE})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()
    # 准确率曲线
    axes[1].plot(range(1, EPOCHS + 1), train_acc_history, marker='o', label='Train Acc')
    axes[1].plot(range(1, EPOCHS + 1), test_acc_history, marker='s', label='Test Acc')
    axes[1].set_title(f'Accuracy Curve ({ACTIVATION_CHOICE} + {OPTIMIZER_CHOICE})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True)
    axes[1].legend()
    
    # 添加训练耗时的文本注释
    fig.suptitle(f'Training Time: {elapsed_time:.1f}s', fontsize=12, color='red')
    
    # 构建文件名
    filename = f"{ACTIVATION_CHOICE}_{OPTIMIZER_CHOICE}_Ghost.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图表已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    losses, train_accs, test_accs, elapsed_time = train()
    plot_and_save(losses, train_accs, test_accs, elapsed_time)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import struct

# ==========================================
# 0. CONFIGURATION
# ==========================================
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# ==========================================
# 1. DATA PREPARATION (LOCAL BINARY FILES)
# ==========================================
def load_mnist_images(file_path):
    """Load MNIST images from binary file"""
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>4I', f.read(16))
        data = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return data

def load_mnist_labels(file_path):
    """Load MNIST labels from binary file"""
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>2I', f.read(8))
        data = np.fromfile(f, dtype=np.uint8)
    return data

class LocalFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_np = self.images[index]
        label = int(self.labels[index])
        
        # Convert to PIL Image
        image = Image.fromarray(image_np, mode='L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load data from local files
print("Loading FashionMNIST data from local files...")
data_dir = 'data/FashionMNIST/raw/'

train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

print(f"Train set size: {len(train_labels)}")
print(f"Test set size: {len(test_labels)}")

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize Datasets
trainset = LocalFashionMNIST(train_images, train_labels, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = LocalFashionMNIST(test_images, test_labels, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# ==========================================
# 2. MODEL 1: MLP (The Baseline)
# ==========================================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 3. MODEL 2: CNN (The Extension)
# ==========================================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv 1: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Conv 2: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_model(model, loader, learning_rate=0.01):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_history = []
    
    print(f"Starting training for {model.__class__.__name__}...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        loss_history.append(avg_loss)
        print(f'   Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')
    return loss_history

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ==========================================
# 5. EXPERIMENTS
# ==========================================
# A. Compare MLP vs CNN
print("\n--- Experiment 1: MLP vs CNN ---")
mlp = MLP()
mlp_losses = train_model(mlp, trainloader)
mlp_acc = evaluate_accuracy(mlp, testloader)
print(f"MLP Final Accuracy: {mlp_acc:.2f}%")

cnn = LeNet()
cnn_losses = train_model(cnn, trainloader)
cnn_acc = evaluate_accuracy(cnn, testloader)
print(f"CNN Final Accuracy: {cnn_acc:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(mlp_losses, label=f'MLP (Acc: {mlp_acc:.1f}%)')
plt.plot(cnn_losses, label=f'CNN (Acc: {cnn_acc:.1f}%)')
plt.title('Training Loss: MLP vs CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plot.png')
plt.show()

# B. Visualize Weights
print("\n--- Experiment 2: Visualizing Weights ---")
def visualize_filters(model):
    filters = model.conv1.weight.data.cpu()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for i, ax in enumerate(axes):
        if i < len(filters):
            ax.imshow(filters[i, 0, :, :], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}')
    plt.suptitle('First Layer Features (Kernels)')
    plt.savefig('cnn_weights.png')
    plt.show()

visualize_filters(cnn)

# C. Learning Rate
print("\n--- Experiment 3: Learning Rates ---")
lrs = [0.1, 0.01, 0.001]
plt.figure(figsize=(10, 5))
for lr in lrs:
    print(f"Testing Learning Rate: {lr}")
    temp_model = MLP()
    losses = train_model(temp_model, trainloader, learning_rate=lr)
    plt.plot(losses, label=f'LR={lr}')
plt.title('Effect of Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('lr_experiment.png')
plt.show()

print("\nAll tasks completed successfully!")
mlp_acc = evaluate_accuracy(mlp, testloader)
print(f"MLP Final Accuracy: {mlp_acc:.2f}%")

# Train CNN
cnn = LeNet()
cnn_losses = train_model(cnn, trainloader)
cnn_acc = evaluate_accuracy(cnn, testloader)
print(f"CNN Final Accuracy: {cnn_acc:.2f}%")

# Plot Comparison
plt.figure(figsize=(10, 5))
plt.plot(mlp_losses, label=f'MLP (Acc: {mlp_acc:.1f}%)')
plt.plot(cnn_losses, label=f'CNN (Acc: {cnn_acc:.1f}%)')
plt.title('Training Loss: MLP vs CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('comparison_plot.png')
plt.show()

# --- Task 2: Visualize Weights ---
print("\n=== TASK 2: Visualizing Learned Weights ===")
def visualize_filters(model):
    # Extract weights from the first layer
    filters = model.conv1.weight.data.cpu()
    # Normalize filters to 0-1 range for plotting
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for i, ax in enumerate(axes):
        if i < len(filters):
            # Plot the i-th filter
            ax.imshow(filters[i, 0, :, :], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}')
    plt.suptitle('First Layer Features (Kernels)')
    plt.savefig('cnn_weights.png')
    plt.show()

visualize_filters(cnn)

# --- Task 3: Learning Rate Schedule ---
print("\n=== TASK 3: Learning Rate Experiment ===")
lrs = [0.1, 0.01, 0.001]
plt.figure(figsize=(10, 5))

for lr in lrs:
    print(f"Testing Learning Rate: {lr}")
    # We use a fresh MLP for each test to be fair
    temp_model = MLP() 
    losses = train_model(temp_model, trainloader, learning_rate=lr)
    plt.plot(losses, label=f'LR={lr}')

plt.title('Effect of Learning Rate on Convergence')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('lr_experiment.png')
plt.show()

print("\nAll tasks completed successfully!")
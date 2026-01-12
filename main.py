import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Course Reference: Lecture 2 (Optimization) & Lecture 4 (CNNs)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# ==========================================
# 1. DATA PREPARATION (Fashion MNIST)
# ==========================================
# Transform: Convert images to tensors and normalize to range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training data
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Download and load test data
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# ==========================================
# 2. THE BASELINE: MLP
# Course Reference: Lecture 3, Slide 6 ("Differentiable MLP")
# ==========================================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Flatten: Converts 2D image (28x28) to 1D vector (784)
        self.flatten = nn.Flatten()
        # Hidden Layer 1: 784 inputs -> 512 neurons
        self.fc1 = nn.Linear(28 * 28, 512)
        # Activation: ReLU (Lecture 3, Slide 9)
        self.relu = nn.ReLU()
        # Hidden Layer 2: 512 -> 256 neurons
        self.fc2 = nn.Linear(512, 256)
        # Output Layer: 10 classes
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 3. THE EXTENSION: CNN (LeNet)
# Course Reference: Lecture 4, Slide 21 ("LeNet Architecture")
# ==========================================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Layer 1: Conv (Slide 7) -> ReLU -> Pooling (Slide 19)
        # Input: 1 channel, Output: 6 feature maps, Kernel: 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        # Average Pooling (Standard for original LeNet, per Lecture 4)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Conv -> ReLU -> Pooling
        # Input: 6 channels, Output: 16 feature maps, Kernel: 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Fully Connected Layers (Slide 2)
        self.flatten = nn.Flatten()
        # Flatten size calculation: 16 channels * 5 * 5 pixels = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # C1 -> S2
        x = self.pool(self.relu(self.conv1(x)))
        # C3 -> S4
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten
        x = self.flatten(x)
        # F5 -> F6 -> Output
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 4. TRAINING & EVALUATION FUNCTIONS
# Course Reference: Lecture 2, Slide 41 ("Training")
# ==========================================
def train_model(model, loader, learning_rate=0.01):
    model = model.to(DEVICE)
    # Loss Function: Cross Entropy (Lecture 2, Slide 34)
    criterion = nn.CrossEntropyLoss()
    # Optimizer: SGD (Lecture 2, Slide 42)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    loss_history = []
    
    print(f"Starting training for {model.__class__.__name__}...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
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
# 5. RUNNING THE HOMEWORK TASKS
# ==========================================

# --- Task 1: Compare MLP vs CNN ---
print("\n=== TASK 1: Compare MLP vs CNN ===")
# Train MLP
mlp = MLP()
mlp_losses = train_model(mlp, trainloader)
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
# Course Reference: Lecture 4, Slide 7 ("Feature Detection")
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
# Course Reference: Lecture 2, Slide 40 ("Gradient Descent Search")
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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. CONFIGURATION
# ==========================================
BATCH_SIZE = 64
EPOCHS = 5  # Kept small for quick testing; increase to 10-20 for better results
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# 1. DATA PREPARATION
# ==========================================
# Transformation: Convert to Tensor and Normalize (Mean=0.5, Std=0.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading FashionMNIST dataset...")
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# ==========================================
# 2. MODEL 1: MLP (The Baseline)
# Course Reference: Lecture 3 (Multilayer Perceptron)
# ==========================================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        # Input: 28x28 = 784 features
        # Hidden Layer 1: 512 neurons
        self.fc1 = nn.Linear(28 * 28, 512)
        # Hidden Layer 2: 256 neurons
        self.fc2 = nn.Linear(512, 256)
        # Output Layer: 10 classes
        self.fc3 = nn.Linear(256, 10)
        # Activation: ReLU (Lecture 3, Slide 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 3. MODEL 2: CNN (The Extension)
# Course Reference: Lecture 4, Slide 21 (LeNet Architecture)
# ==========================================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Layer 1: Convolution -> ReLU -> Pooling
        # Input: 1 channel (grayscale), Output: 6 feature maps
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Convolution -> ReLU -> Pooling
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # C1 -> S2 (Pooling)
        x = self.pool(self.relu(self.conv1(x)))
        # C3 -> S4 (Pooling)
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten for Dense Layers
        x = self.flatten(x)
        # F5 -> F6 -> Output
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_model(model, learning_rate=0.01):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        print(f'   Epoch {epoch+1}: Loss = {avg_loss:.4f}')
        
    return loss_history

def evaluate_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ==========================================
# 5. EXPERIMENTS & VISUALIZATION
# ==========================================

# --- A. Compare MLP vs CNN ---
print("\n--- Experiment 1: MLP vs CNN ---")

print("Training MLP (Baseline)...")
mlp = MLP()
mlp_losses = train_model(mlp)
mlp_acc = evaluate_accuracy(mlp)
print(f"MLP Accuracy: {mlp_acc:.2f}%")

print("\nTraining CNN (LeNet Extension)...")
cnn = LeNet()
cnn_losses = train_model(cnn)
cnn_acc = evaluate_accuracy(cnn)
print(f"CNN Accuracy: {cnn_acc:.2f}%")

# Plot Comparison
plt.figure(figsize=(10, 5))
plt.plot(mlp_losses, label='MLP Loss')
plt.plot(cnn_losses, label='CNN Loss')
plt.title('Training Loss: MLP vs CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('mlp_vs_cnn_loss.png')
plt.show()

# --- B. Visualize Weights (Extension 2) ---
# Course Reference: Lecture 4, Slide 7 (Features/Filters)
print("\n--- Experiment 2: Visualizing Weights ---")
def visualize_kernels(model):
    # Extract weights from the first convolutional layer
    kernels = model.conv1.weight.data.cpu()
    
    # Normalize for visualization (0 to 1 range)
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    
    # Plot first 6 filters
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for i, ax in enumerate(axes):
        if i < len(kernels):
            ax.imshow(kernels[i, 0, :, :], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}')
    plt.suptitle('Learned Kernels (Features) of Layer 1')
    plt.savefig('cnn_kernels.png')
    plt.show()

visualize_kernels(cnn)

# --- C. Learning Rate Schedule (Extension 3) ---
# Course Reference: Lecture 2, Slide 40 (Step Size/Learning Rate)
print("\n--- Experiment 3: Learning Rates ---")
lrs = [0.1, 0.01, 0.001]
plt.figure(figsize=(10, 5))

for lr in lrs:
    print(f"Testing Learning Rate: {lr}")
    temp_model = MLP() # Use MLP for quick testing
    losses = train_model(temp_model, learning_rate=lr)
    plt.plot(losses, label=f'LR={lr}')

plt.title('Effect of Learning Rate on Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('lr_comparison.png')
plt.show()

print("\nAll experiments finished!")
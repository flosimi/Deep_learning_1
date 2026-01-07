import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
# Device configuration (use GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5  # Keep it low for demonstration speed (increase to 10-20 for better results)

# Transformations: Convert to Tensor and Normalize to range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])# Data Loading
print("\nDownloading and Loading Data...")
# Training Data
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
                                                  transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test Data
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, 
                                                 transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# ==========================================
# 2. Model Definitions
# ==========================================

# --- Model A: Simple MLP (Baseline) ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Input: 28x28 = 784 pixels
        # Hidden Layer: 256 neurons
        # Output: 10 classes
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10) 

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out

# --- Model B: CNN (Extension 1) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv Layer 1: Input 1 channel (grayscale), Output 32 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Image becomes 14x14
        )
        # Conv Layer 2: Input 32 channels, Output 64 channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # Image becomes 7x7
        )
        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 7 * 7, 600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

# ==========================================
# 3. Training & Evaluation Engine
# ==========================================
def train_model(model, train_loader, learning_rate=0.001, use_scheduler=False):
    model = model.to(device)
    model.train()  # Explicitly set to training mode
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Extension 3: Learning Rate Scheduler
    scheduler = None
    if use_scheduler:
        # Reduces LR by factor of 0.1 every 3 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    loss_history = []
    
    print(f"Training {model.__class__.__name__}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, LR: {current_lr}")
        
        if scheduler:
            scheduler.step()
            
    return loss_history

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    model.train()  # Set back to training mode
    return acc

# ==========================================
# 4. Running the Comparisons (Extension 1)
# ==========================================

# 1. Train MLP
mlp_model = SimpleMLP()
mlp_losses = train_model(mlp_model, train_loader)
mlp_acc = evaluate_model(mlp_model, test_loader)
torch.save(mlp_model.state_dict(), 'mlp_model.pth')
print("MLP model saved to 'mlp_model.pth'")

# 2. Train CNN
cnn_model = SimpleCNN()
cnn_losses = train_model(cnn_model, train_loader)
cnn_acc = evaluate_model(cnn_model, test_loader)
torch.save(cnn_model.state_dict(), 'cnn_model.pth')
print("CNN model saved to 'cnn_model.pth'")

print("\n--- Extension 1: Performance Comparison ---")
print(f"MLP Accuracy: {mlp_acc:.2f}%")
print(f"CNN Accuracy: {cnn_acc:.2f}%")

# Plotting Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(mlp_losses, label='MLP Loss')
plt.plot(cnn_losses, label='CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss: MLP vs CNN')
plt.legend()
plt.show()

# ==========================================
# 5. Extension 2: Visualize Hidden Layer Weights
# ==========================================
def visualize_mlp_weights(model):
    print("\n--- Extension 2: Visualizing MLP Weights ---")
    # Extract weights from the first layer (fc1)
    # Shape is [256, 784] -> 256 neurons, each with 784 weights mapping to pixels
    weights = model.fc1.weight.data.cpu().numpy()
    
    # Plot the first 16 neurons' weights as images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('Learned Weights of First Hidden Layer (MLP)')
    
    for i, ax in enumerate(axes.flat):
        # Reshape 784 back to 28x28
        weight_img = weights[i].reshape(28, 28)
        # Use a diverging colormap (red=positive, blue=negative)
        ax.imshow(weight_img, cmap='seismic') 
        ax.axis('off')
        
    plt.show()

visualize_mlp_weights(mlp_model)

# ==========================================
# 6. Extension 3: Learning Rate Experiment
# ==========================================
print("\n--- Extension 3: Learning Rate Experiments (using CNN) ---")
lrs = [0.001, 0.0001] # Standard vs Low LR
results = {}

for lr in lrs:
    print(f"\nTesting Learning Rate: {lr}")
    temp_model = SimpleCNN() # Create fresh model
    losses = train_model(temp_model, train_loader, learning_rate=lr, use_scheduler=True)
    acc = evaluate_model(temp_model, test_loader)
    results[lr] = acc
    print(f"LR {lr} Final Accuracy: {acc:.2f}%")

print("\nFinal Results Summary:", results)
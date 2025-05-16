# CIFAR-10 Image Classification Mini Project

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = 30

# Create results directory if it doesn't exist
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/'))
os.makedirs(results_dir, exist_ok=True)

# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

# Train-Validation Split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Data loading completed!")

# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Check if the baseline model exists
model_path = '../models/baseline_model.pth'

# Ensure the models directory exists
models_dir = os.path.dirname(model_path)
os.makedirs(models_dir, exist_ok=True)

if not os.path.exists(model_path):
    print("Baseline model not found. Training a new model...")
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation Loss
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for val_images, val_labels in val_loader:
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
        model.train()
        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        training_losses.append(epoch_train_loss)
        validation_losses.append(epoch_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    # Save loss curves
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()
    torch.save(model.state_dict(), model_path)
    print("Baseline model trained and saved!")
else:
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    print("Baseline model loaded!")

# Experiment 1: Random Label Shuffle
random_label_dataset = [(img, random.randint(0, 9)) for img, _ in train_subset]
random_label_loader = DataLoader(random_label_dataset, batch_size=batch_size, shuffle=True)
print("Random Label Shuffle dataset prepared!")

# Experiment 2: Label Noise (20%)
noisy_label_dataset = list(train_subset)
noise_indices = np.random.choice(len(noisy_label_dataset), int(0.2 * len(noisy_label_dataset)), replace=False)
for idx in noise_indices:
    img, label = noisy_label_dataset[idx]
    noisy_label_dataset[idx] = (img, random.randint(0, 9))
noisy_label_loader = DataLoader(noisy_label_dataset, batch_size=batch_size, shuffle=True)
print("Label Noise (20%) dataset prepared!")

# Experiment 3: Input Perturbation (e.g., Gaussian Noise)
def add_noise(img):
    noise = torch.randn_like(img) * 0.1
    return torch.clamp(img + noise, 0, 1)

def perturb_collate_fn(batch):
    images, labels = zip(*batch)
    noisy_images = [add_noise(img) for img in images]
    return torch.stack(noisy_images), torch.tensor(labels)

perturbed_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=perturb_collate_fn)
print("Input Perturbation dataset prepared!")

# Evaluation and Confusion Matrix
def evaluate_and_visualize(model, data_loader, name, show_plot=True):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plot_path = os.path.join(results_dir, f"{name}_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Confusion matrix saved at: {plot_path}")
    if show_plot:
        plt.show()
    plt.close()
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# Evaluate and visualize all datasets
evaluate_and_visualize(model, test_loader, "Baseline_Test", show_plot=True)
evaluate_and_visualize(model, random_label_loader, "Random_Label_Shuffle", show_plot=True)
evaluate_and_visualize(model, noisy_label_loader, "Label_Noise_20", show_plot=True)
evaluate_and_visualize(model, perturbed_loader, "Input_Perturbation", show_plot=True)

print("All experiments completed!")

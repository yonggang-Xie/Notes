# vit_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader

# Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ViT's expected input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the Vision Transformer Model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10)
model.to('cuda')

# Set Up Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        
        outputs = model(images).logits
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    pass

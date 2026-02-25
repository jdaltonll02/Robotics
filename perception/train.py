"""
PyTorch training script for semantic segmentation or depth estimation.
Collects data from simulation, trains a CNN, and saves the model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SegmentationNet
from dataset import SimulationDataset

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3

# Dataset and DataLoader
train_dataset = SimulationDataset('data/images', 'data/labels')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = SegmentationNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), 'model/segmentation_net.pth')

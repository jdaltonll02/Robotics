"""
Train PPO with CNN feature extractor and TensorBoard logging.
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from agents.ppo_cnn import PPOActorCritic

OBS_SHAPE = (3, 64, 64)  # Example: 3-channel 64x64 image
ACTION_DIM = 2
LR = 3e-4
EPOCHS = 1000

model = PPOActorCritic(OBS_SHAPE, ACTION_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter('runs/ppo_cnn_experiment')

for epoch in range(EPOCHS):
    obs = torch.randn(1, *OBS_SHAPE)  # Replace with real env obs
    action, value = model(obs)
    # Compute PPO loss, update, log metrics...
    writer.add_scalar('Loss/ppo', torch.rand(1).item(), epoch)  # Dummy
    print(f"Epoch {epoch+1}/{EPOCHS}")

# Save model
torch.save(model.state_dict(), 'ppo_cnn.pth')
writer.close()

import pytest
import torch
from rl_control.ppo import ActorCritic

def test_actor_critic_forward():
    model = ActorCritic(state_dim=100, action_dim=2)
    x = torch.randn(100)
    action, value = model(x)
    assert action.shape[-1] == 2
    assert value.shape[-1] == 1

import torch
import torch.nn as nn
import numpy as np

obs_dim = 100
act_dim = 6

# make a simple mlp policy
pi = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 32),
    nn.Tanh(),
    nn.Linear(32, act_dim)
)

# predict action from batch of observations
obs = np.random.random((64, obs_dim))
obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
actions = pi(obs_tensor).cpu()
actions = actions.detach().numpy()

# check shape
print(obs.shape)
print(actions.shape)


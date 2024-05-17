
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque
import rl_baseline_framework

import random

class CNNBase(nn.Module):
    def __init__(self):
        super(CNNBase, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.flattened_dim = 16 * 7 * 7  # Adjust according to input size and network architecture

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return x

class CNNActor(nn.Module):
    def __init__(self):
        super(CNNActor, self).__init__()
        self.cnn_base = CNNBase()
        self.fc1 = nn.Linear(self.cnn_base.flattened_dim, 64)
        self.fc2 = nn.Linear(64, 1)  # Output dimension should match the action space

    def forward(self, x):
        x = self.cnn_base(x)
        x = F.relu(self.fc1(x))
        action = torch.tanh(self.fc2(x))  # Assuming action space is [-1, 1]
        return action

class CNNCritic(nn.Module):
    def __init__(self):
        super(CNNCritic, self).__init__()
        self.cnn_base = CNNBase()
        self.fc1 = nn.Linear(self.cnn_base.flattened_dim + 1, 64)  # +1 for action input
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, action):
        x = self.cnn_base(x)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value
    

class DDPGAgent():
    def __init__(self, actor, critic, actor_target, critic_target, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=1e-3):
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        return action.flatten()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update critic
        next_actions = self.actor_target(next_states)
        next_Q_values = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * next_Q_values * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
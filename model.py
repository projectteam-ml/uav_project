from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from helpers import SinusoidalPosEmb
# from tianshou.data import Batch, ReplayBuffer, to_torch

class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Tanh if activation == 'mish' else nn.ReLU
        # _act = nn.Tanh if activation == 'mish' else nn.Tanh
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):
        # print("state:",state)
        state = state.float()
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        # print(x.shape,time.shape,processed_state.shape)
        x = torch.cat([x, t, processed_state], dim=1)
        # print(x.shape)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        # return torch.tanh(x)
        return x

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([torch.tensor(v).flatten() for v in
                                   self.parameters()]))

class TransformerDenoiser(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.time_embed = nn.Linear(1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, t, s):
        t = t.unsqueeze(-1).float()  # [batch, 1]
        x_embed = self.action_embed(x)
        s_embed = self.state_embed(s)
        t_embed = self.time_embed(t)

        # (batch, seq=3, dim)
        sequence = torch.stack([x_embed, s_embed, t_embed], dim=1)
        output = self.transformer(sequence)

        return torch.tanh(self.output(output[:, 0, :]))


class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        _act = nn.Tanh if activation == 'mish' else nn.ReLU
        # _act = nn.Tanh if activation == 'mish' else nn.Tanh
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.q1_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
    # def forward(self, obs):
    #     return self.q1_net(obs), self.q2_net(obs)
    #
    # def q_min(self, obs):
    #     return torch.min(*self.forward(obs))
    def forward(self, state, action):
        # state = to_torch(state, device='cpu', dtype=torch.float32)
        # action = to_torch(action, device='cpu', dtype=torch.float32)
        state = state.float()
        action = action.float()
        processed_state = self.state_mlp(state)
        x = torch.cat([processed_state, action], dim=-1)
        return self.q1_net(x)

    def q_min(self, obs, action):
        # obs = to_torch(obs, device='cuda:0', dtype=torch.float32)
        # action = to_torch(action, device='cuda:0', dtype=torch.float32)
        return torch.min(*self.forward(obs, action))

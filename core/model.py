import numpy as np
from random import random, randrange

import torch
import torch.nn as nn


# original
class CNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNDQN, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)

    def act(self, state, epsilon, device):
        if random() > epsilon:
            state = torch.FloatTensor(np.float32(state)) \
                .unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = randrange(self._num_actions)
        return action


# Self-Attention
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


# class CNNDQN(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(CNNDQN, self).__init__()
#         self._input_shape = input_shape
#         self._num_actions = num_actions
#
#         self.features = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#
#         self.self_attention = SelfAttention(64, heads=8)
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.feature_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_actions)
#         )
#
#     def forward(self, x):
#         x = self.features(x).view(x.size()[0], -1)
#         x = self.self_attention(x, x, x, None)
#         return self.fc(x)
#
#     @property
#     def feature_size(self):
#         x = self.features(torch.zeros(1, *self._input_shape))
#         return x.view(1, -1).size(1)
#
#     def act(self, state, epsilon, device):
#         if random() > epsilon:
#             state = torch.FloatTensor(np.float32(state)) \
#                 .unsqueeze(0).to(device)
#             q_value = self.forward(state)
#             action = q_value.max(1)[1].item()
#         else:
#             action = randrange(self._num_actions)
#         return action



# H-DQN
class HierarchicalDQN(nn.Module):
    def __init__(self, high_level_input_shape, low_level_input_shape, num_actions):
        super(HierarchicalDQN, self).__init__()

        # High-level policy network
        self.high_level_features = nn.Sequential(
            nn.Linear(high_level_input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Low-level policy network
        self.low_level_features = nn.Sequential(
            nn.Conv2d(low_level_input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(128 + 64, 512),
            nn.ReLU()
        )

        # Q-values output layer
        self.q_values = nn.Linear(512, num_actions)

    def forward(self, high_level_state, low_level_state):
        high_level_features = self.high_level_features(high_level_state)
        low_level_features = self.low_level_features(low_level_state)

        # Concatenate high-level and low-level features
        combined_features = torch.cat([high_level_features, low_level_features], dim=1)

        # Pass through shared layers
        shared_output = self.shared_fc(combined_features)

        # Q-values output
        q_values = self.q_values(shared_output)
        return q_values

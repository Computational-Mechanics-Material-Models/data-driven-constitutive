#!/usr/bin/env python

# Artificial Neural Network models for constitutive modeling of materials

import torch
import torch.nn as nn
import torch.nn.functional as F

# LSTM Architecture from Logarzo et al. 2021 https://doi.org/10.1016/J.CMA.2020.113482
class logarzo_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout1, dropout2):
        super(logarzo_lstm, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_dim2, 6)  # Output layer for regression # TODO think of making this generic, not hardcoded as 6

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x) # Select last timestep for prediction
        return x


# Logarzo architecture with GRU units instead of LSTM units
class logarzo_gru(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout1, dropout2):
        super(logarzo_gru, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_dim2, 6)  # Output layer for regression

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = self.fc(x)
        return x


# Simple feed-forward network with linear layers for time-independent model.
# Input: current strain, current stress, strain increment
# Output: stress increment
class incremental_linear(nn.Module):
    def __init__(self, input_dim, num_hidden, hidden_dim, output_dim, hidden_activation):
        super(incremental_linear, self).__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.output_dim = output_dim
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_hidden-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        match hidden_activation:
            case 'relu':
                self.fh = F.relu
            case 'tanh':
                self.fh = F.tanh
            case 'sigmoid':
                 self.fh = F.sigmoid

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.fh(layer(x))
        x = self.layers[-1](x) # Last layer activation = identity
        return x
    




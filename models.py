#!/usr/bin/env python

# Artificial Neural Network models for constitutive modeling of materials

import torch.nn as nn

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


# Logarzo architecture with GRU units instead of LSTM
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


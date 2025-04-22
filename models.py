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
class ff_linear(nn.Module):
    def __init__(self, input_dim, num_hidden, hidden_dim, output_dim, hidden_activation):
        super(ff_linear, self).__init__()
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
            case 'leaky_relu':
                self.fh = F.leaky_relu # Default negative slope of 0.01

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.fh(layer(x))
        x = self.layers[-1](x) # Last layer activation = identity
        return x
    

# Recursive neural network with linear layers for time-independent model.
# Input: previous strain, strain increment, previous stress
# Output: stress increment, fed back recursively to the previous stress
class rnn_linear(nn.Module):
    def __init__(self, input_dim, num_hidden, hidden_dim, output_dim, hidden_activation):
        super(rnn_linear, self).__init__()
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
            case 'leaky_relu':
                self.fh = F.leaky_relu # Default negative slope of 0.01

    def forward(self, x): # TODO: there is likely  a better way of doing this, e.g. with stress as hidden variable
        # Assumes x shape is [batch_size, sequence_length, features (18)] with stress at the end
        if (self.training):
            # Train model using stress in the data as input
            # This is not recursive and should be fast and accurate
            stress_prev = x[:, :, 12:]
            for layer in self.layers[:-1]:
                x = self.fh(layer(x))
            stress_incr = self.layers[-1](x) # Last layer activation = identity
            stress = stress_prev + stress_incr
        else:
            # Evaluate model using its own recursive prediction of the stress
            stress = torch.zeros(x.shape[0], x.shape[1], self.output_dim)
            stress_prev_t = x[:, 0, 12:]
            for t in range(x.shape[1]):
                x_t = x[:,t,:]
                x_t[:,12:] = stress_prev_t
                for layer in self.layers[:-1]:
                    x_t = self.fh(layer(x_t))
                stress_incr_t = self.layers[-1](x_t) # Last layer activation = identity
                stress[:, t, :] = stress_prev_t + stress_incr_t
                stress_prev_t = stress[:, t, :]
        return stress

# Recursive Neural Network architecture of Bhattacharya et al. 2023. https://doi.org/10.1137/22M1499200
# Uses 2 feed-forward neural network:
# the first one computes the derivatives of the state variables
# the second one compute the stress from updated state variables
class bhattacharya_rnn(nn.Module):
    def __init__(self, num_statevar, num_hidden_G, hidden_dim_G, activation_G,
                                     num_hidden_F, hidden_dim_F, activation_F):
        super(bhattacharya_rnn, self).__init__()
        self.num_statevar = num_statevar
        input_dim_G = 6 + num_statevar # input = strain (6) and state variables
        output_dim_G = num_statevar # output = state variables
        input_dim_F = 6 + 6 + num_statevar # input = strain (6), strain derivatives (6) and state variables
        output_dim_F = 6 # output = stress

        networks = [nn.ModuleList(), nn.ModuleList()]
        activations = [None, None]
        for n, (input_dim, num_hidden, hidden_dim, output_dim, activation) in enumerate(zip((input_dim_G, input_dim_F),
                                                                                            (num_hidden_G, num_hidden_F),
                                                                                            (hidden_dim_G, hidden_dim_F),
                                                                                            (output_dim_G, output_dim_F),
                                                                                            (activation_G, activation_F))):
            networks[n].append(nn.Linear(input_dim, hidden_dim))
            for i in range(num_hidden-1):
                networks[n].append(nn.Linear(hidden_dim, hidden_dim))
            networks[n].append(nn.Linear(hidden_dim, output_dim))

            match activation:
                case 'selu':
                    activations[n] = F.selu # Function used in the paper
                case 'relu':
                    activations[n] = F.relu
                case 'tanh':
                    activations[n] = F.tanh
                case 'sigmoid':
                      activations[n] = F.sigmoid
                case 'leaky_relu':
                    activations[n] = F.leaky_relu # Default negative slope of 0.01
        self.network_G = networks[0]
        self.network_F = networks[1]
        self.f_G = activations[0]
        self.f_F = activations[1]

    def forward(self, x):
        # Assumes x shape is [batch_size, sequence_length, features (13)]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # Assume features is (time, strain, rate of deformation)
        timesteps = x[:, 1:, 0] - x[:, :-1, 0]
        strain = x[:, :, 1:7]
        strain_dot = x[:, :, 7:]

        # First network G: state variable evolution
        # We don't use torch.nn.RNN (which would likely be more practical) because activations are only tanh or reLU
        statevar = torch.zeros(batch_size, seq_len, self.num_statevar)
        for i in range(seq_len - 1):
            dt = timesteps[:, i]
            x_t = torch.cat([strain[:, i, :], statevar[:, i, :]], dim = 1)
            for layer in self.network_G[:-1]:
                x_t = self.f_G(layer(x_t))
            statevar_dot = self.network_G[-1](x_t) # Last layer activation = identity
            statevar[:, i+1, :] = statevar[:, i, :] + dt.unsqueeze(1) * statevar_dot # Forward Euler update

        # Second network F: stress calculation
        x_F = torch.cat([strain, strain_dot, statevar], dim = 2)
        for layer in self.network_F[:-1]:
            x_F = self.f_F(layer(x_F))
        x_F = self.network_F[-1](x_F) # Last layer activation = identity

        return x_F
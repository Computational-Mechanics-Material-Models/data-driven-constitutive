#!/usr/bin/env python

# BSD 3-Clause License
#
# Copyright (c) 2025, Cusatis Computational Services, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
# Network predicts stress increment ds and updates stress: s_t = s_(t-1) + ds
class ff_linear(nn.Module):
    # For constitutive modeling:
    # Input: previous strain, current strain, previous stress
    # Output: stress increment
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
class rnn_linear(nn.Module):
    # For constitutive modeling, network predicts stress increment ds from
    # previous strain e_(t-1), current strain e_t, previous stress s_(t-1), and
    # performs forward update to current stress: s_t = s_(t-1) + ds
    #
    # By design, this model cannot capture behavior such as elastic damage.
    # That is because for a given state of stress and strain, the prediction
    # only depends on the next strain value, e.g., if you unload to zero and reload,
    # the model will not predict a smaller (damaged) elastic modulus in the realoading
    # because it lacks state variables to inform it that this unloaded state
    # different from the initial never-loaded state
    def __init__(self, num_hidden, size_hidden, activation, training_style):
        super(rnn_linear, self).__init__()
        self.num_hidden = num_hidden
        self.size_hidden = size_hidden
        self.f = activation
        self.training_style = training_style # 'direct', or 'recursive'
        self.size_sym_tensor = 6
        # Input layer: separate inputs into 3 independent partial layers
        # Makes stress as hidden variable easier than single size-18 layer
        # Only one set of bias necessary for all 3 additive partial layers
        self.input_layers = nn.ModuleList()
        self.input_layers.append(nn.Linear(self.size_sym_tensor, size_hidden, bias = True)) # Previous strain
        self.input_layers.append(nn.Linear(self.size_sym_tensor, size_hidden, bias = False)) # Current strain
        self.input_layers.append(nn.Linear(self.size_sym_tensor, size_hidden, bias = False)) # Previous stress
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden - 1):
            self.hidden_layers.append(nn.Linear(size_hidden, size_hidden))
        # Output layer
        self.output_layer = nn.Linear(size_hidden, self.size_sym_tensor)

    def set_training_style(self, training_style):
        self.training_style = training_style

    # The extra argument is the entire stress history for direct training
    # and the first value only [batch_size, 1, 6] for recursive training
    def forward(self, strain_history, stress_history):
        if (self.training and self.training_style == 'direct' and stress_history.shape[1] != strain_history.shape[1]):
            raise ValueError("training with direct training requires extra variable strain_history for the entire sequence")
        # Assumes strain_history shape [batch_size, sequence_length, features]
        batch_size = strain_history.shape[0]
        seq_len = strain_history.shape[1]
        strain_prev = strain_history[:, :-1, :]
        strain_curr = strain_history[:, 1:, :]
        stress_curr = torch.zeros(batch_size, seq_len, self.size_sym_tensor)
        stress_curr[:, 0, :] = stress_history[:, 0, :]
        if (self.training and self.training_style == 'direct'):
            # Train model using stress history in the data as input
            stress_prev = stress_history[:, :-1, :]
            x = self.f(self.input_layers[0](strain_prev) +
                       self.input_layers[1](strain_curr) +
                       self.input_layers[2](stress_prev))
            for layer in self.hidden_layers:
                x = self.f(layer(x))
            stress_incr = self.output_layer(x) # Last layer activation = identity
            stress_curr[:, 1:, :] = stress_prev + stress_incr
        else:
            # Train/Evaluate model on its own recursive prediction of stress
            stress_prev_t = stress_history[:, 0, :]
            for t in range(seq_len - 1):
                strain_prev_t = strain_prev[:, t, :]
                strain_curr_t = strain_curr[:, t, :]
                x_t = self.f(self.input_layers[0](strain_prev_t) +
                             self.input_layers[1](strain_curr_t) +
                             self.input_layers[2](stress_prev_t))
                for layer in self.hidden_layers:
                    x_t = self.f(layer(x_t))
                stress_incr_t = self.output_layer(x_t) # Last layer activation = identity
                stress_curr_t = stress_prev_t + stress_incr_t
                stress_curr[:, t+1, :] = stress_curr_t
                stress_prev_t = stress_curr_t
        return stress_curr

# Recursive Neural Network architecture of Bhattacharya et al. 2023. https://doi.org/10.1137/22M1499200
# TODO: make statevar a proper hidden variable
class bhattacharya_rnn(nn.Module):
    # Uses 2 feed-forward neural network for constitutive modeling:
    # First network G recursively computes derivatives of state variables xidot_t from
    # the current strain e_t, current value of state variable xi_t and time
    # increment dt and updates it with Forward Euler xi_(t+1) = xi_t + xidot_t * dt
    # Second network F computes the stress from the current strain e_t, current
    # strain derivative edot_t and current state variable xi_t
    #
    # By design, the derivative of state variable is independent of the loading
    # direction, which seems wrong. i.e. state variables evolve identically
    # regardless of the loading increment
    # Additionally for rate-independent models, getting the time-derivative and
    # updating would not work since the strain rate does not matter, only the
    # strain increment.
    def __init__(self, num_statevar, num_hidden_G, size_hidden_G, activation_G,
                                     num_hidden_F, size_hidden_F, activation_F):
        super(bhattacharya_rnn, self).__init__()
        self.num_statevar = num_statevar
        self.size_sym_tensor = 6
        # First network G:
        # Input layer: separate inputs into 2 independent partial layers
        # State variables as hidden variables. Only one set of bias necessary
        self.G_input_layers = nn.ModuleList()
        self.G_input_layers.append(nn.Linear(self.size_sym_tensor, size_hidden_G, bias = True)) # Current strain
        self.G_input_layers.append(nn.Linear(self.num_statevar, size_hidden_G, bias = False)) # State Variables

        size_input_G = self.size_sym_tensor + num_statevar # input = strain (6) and state variables
        size_output_G = num_statevar # output = state variables
        size_input_F = self.size_sym_tensor + self.size_sym_tensor + num_statevar # input = strain (6), strain derivatives (6) and state variables
        size_output_F = self.size_sym_tensor # output = stress

        networks = [nn.ModuleList(), nn.ModuleList()]
        activations = [None, None]
        for n, (size_input, num_hidden, size_hidden, output_dim, activation) in enumerate(zip((size_input_G, size_input_F),
                                                                                              (num_hidden_G, num_hidden_F),
                                                                                              (size_hidden_G, size_hidden_F),
                                                                                              (size_output_G, size_output_F),
                                                                                              (activation_G, activation_F))):
            networks[n].append(nn.Linear(size_input, size_hidden))
            for i in range(num_hidden-1):
                networks[n].append(nn.Linear(size_hidden, size_hidden))
            networks[n].append(nn.Linear(size_hidden, output_dim))

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

    def forward(self, x): # State variable always initialized at zero so no need to pass it as extra arg
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
            x_t = torch.cat([strain[:, i, :], statevar[:, i, :]], dim = 1) # TODO, I don't want to have to do that and should make a multi-first layer with single bias like RNN
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


# Neural network architecture based on isotropic tangent stiffness
# TODO: add state variables after it is tested on the elastic part
# TODO: all my nuilt-in RNN start looking the same with the training style,
# direct training vs recursive evaluation etc
# think about building a parent class eventually to avoid duplicating code, or
# deleting models that don't work well as you continue development and testing
class rnn_tangent_iso(nn.Module):
    # For constitutive modeling, network predicts isotropic tangent stiffness K
    # K = \lambda \mathbf {I} \otimes \mathbf {I} +2\mu {\mathsf {I}}} from
    # previous strain e_(t-1), current strain e_t, previous stress s_(t-1), and
    # performs forward stres update: s_t = s_(t-1) + K : (e_t - e_(t-1))
    def __init__(self, num_hidden, size_hidden, activation, training_style):
        super(rnn_tangent_iso, self).__init__()
        self.num_hidden = num_hidden
        self.size_hidden = size_hidden
        self.f = activation
        self.training_style = training_style # 'direct', or 'recursive'
        self.size_sym_tensor = 6
        self.size_tangent_operator = 2
        # Input layer: separate inputs into 3 independent partial layers
        # Make stress hidden and use only one set of bias
        self.input_layers = nn.ModuleList()
        self.input_layers.append(nn.Linear(self.size_sym_tensor, size_hidden, bias = True)) # Previous strain
        self.input_layers.append(nn.Linear(self.size_sym_tensor, size_hidden, bias = False)) # Current strain
        self.input_layers.append(nn.Linear(self.size_sym_tensor, size_hidden, bias = False)) # Previous stress
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden - 1):
            self.hidden_layers.append(nn.Linear(size_hidden, size_hidden))
        # Output layer
        self.output_layer = nn.Linear(size_hidden, self.size_tangent_operator)

    # Turn the vectorized tangent Lame parameters into the 6x6 stiffness matrix
    def Klame_to_K66(self, Klame):
        shapeK66 = list(Klame.shape)[:-1]
        shapeK66 += [6, 6]
        K66 = torch.zeros(shapeK66)
        lambd = Klame[..., 0]
        twomu = 2.0 * Klame[..., 1]
        for i in range(3):
            K66[..., i, i] = lambd + twomu
            K66[..., i+3, i+3] = twomu
            K66[..., i%3, (i+1)%3] = K66[..., (i+1)%3, i%3] = lambd
        return K66

    def set_training_style(self, training_style):
        self.training_style = training_style

    # The extra argument is the entire stress history for direct training
    # and the first value only [batch_size, 1, 6] for recursive training/eval
    def forward(self, strain_history, stress_history):
        if (self.training and self.training_style == 'direct' and stress_history.shape[1] != strain_history.shape[1]):
            raise ValueError("training with direct training requires extra variable strain_history for the entire sequence")
        # Assumes strain_history shape [batch_size, sequence_length, features (18)]
        batch_size = strain_history.shape[0]
        seq_len = strain_history.shape[1]
        strain_prev = strain_history[:, :-1, :]
        strain_curr = strain_history[:, 1:, :]
        strain_incr = strain_curr - strain_prev
        stress_curr = torch.zeros(batch_size, seq_len, self.size_sym_tensor)
        stress_curr[:, 0, :] = stress_history[:, 0, :]
        if (self.training and self.training_style == 'direct'):
            # Train model using stress history in the data as input
            stress_prev = stress_history[:, :-1, :]
            x = self.f(self.input_layers[0](strain_prev) +
                       self.input_layers[1](strain_curr) +
                       self.input_layers[2](stress_prev))
            for layer in self.hidden_layers:
                x = self.f(layer(x))
            Klame = self.output_layer(x) # Last layer activation = identity
            K = self.Klame_to_K66(Klame)
            stress_curr[:, 1:, :] = stress_prev + torch.matmul(K, strain_incr.unsqueeze(-1)).squeeze(-1) # unsqueeze / squeeze to batch multiply
        else:
            # Train/Evaluate model using its own recursive prediction of the stress
            stress_prev_t = stress_history[:, 0, :]
            for t in range(seq_len - 1):
                strain_prev_t = strain_prev[:, t, :]
                strain_curr_t = strain_curr[:, t, :]
                strain_incr_t = strain_incr[:, t, :]
                x_t = self.f(self.input_layers[0](strain_prev_t) +
                             self.input_layers[1](strain_curr_t) +
                             self.input_layers[2](stress_prev_t))
                for layer in self.hidden_layers:
                    x_t = self.f(layer(x_t))
                Klame_t = self.output_layer(x_t) # Last layer activation = identity
                K_t = self.Klame_to_K66(Klame_t)
                stress_curr_t = stress_prev_t + torch.matmul(K_t, strain_incr_t.unsqueeze(-1)).squeeze(-1)  # unsqueeze / squeeze to batch multiply
                stress_curr[:, t+1, :] = stress_curr_t
                stress_prev_t = stress_curr_t
        return stress_curr


#!/usr/bin/env python

# Training utilities for Artificial Neural Networks for constitutive modeling of materials
# Loss functions and network training routines
# Optuna hyperparameters training

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset


# Loss function of Yuhui Lyu
def loss_yuhuilyu(model, X_batch, y_batch, extra_args_list_batch, R):
    if (extra_args_list_batch == None):
        y_pred = model(X_batch)
    else:
        y_pred = model(X_batch, *extra_args_list_batch)

    strain = X_batch.to(dtype=torch.float32) # TODO: I don't think this is necessary. Default should be torch.float32

    # Compute Term 1: MSE loss between predictions and ground truth
    term1 = torch.mean(torch.sum((y_pred - y_batch) ** 2, dim=[1, 2])) # TODO: mean over dim 1,2 divides by 6 which is not in the formula on Overleaf that uses the L2 norm

    # # Compute Term 2: Rotation-based transformation
    # # TODO JBC: this makes no sense ! The whole matrix should be rotated, not just the diagonal...
    # # Convert R to a PyTorch tensor
    # R_tensor = torch.tensor(R, dtype=torch.float32, device=y_pred.device)
    # # Extract the diagonal strain components (first three features)
    # diagonal_strain = strain[:, :, :3]  # Shape: (batch_size, 1000, 3)
    # rotated_strain = torch.matmul(diagonal_strain.view(-1, 3), R_tensor)
    # rotated_strain = rotated_strain.view(diagonal_strain.shape)  # Reshape to original shape
    # # Compute R^{-1} (inverse of R)
    # R_tensor_inv = torch.linalg.inv(R_tensor)
    # transformed_strain = torch.matmul(rotated_strain.view(-1, 3), R_tensor_inv)
    # transformed_strain = transformed_strain.view(rotated_strain.shape)
    # # Pad transformed strain to match the input shape
    # transformed_strain_padded = torch.cat(
    #     [transformed_strain, torch.zeros_like(strain[:, :, 3:])], dim=-1 # TODO: adding zero extradiagonal is wrong here too
    # )
    # # Predict stress using the model
    # predicted_transformed_stress = model(transformed_strain_padded)
    # # Apply rotation matrix to stress # TODO: same here, rotating the diagonal is wrong
    # rotated_stress = torch.matmul(y_pred[:, :, :3].reshape(-1, 3), R_tensor)
    # rotated_stress = rotated_stress.view(y_pred[:, :, :3].shape)
    # # Compute Term 2 difference
    # difference = predicted_transformed_stress[:, :, :3] - rotated_stress # TODO: same here, comparing only the diagonal is wrong
    # term2 = torch.mean(torch.sum(difference ** 2, dim=[1, 2]))

    # Compute Term 3: Delta stress change
    strain_current = strain[:, 1:, :6]
    strain_previous = strain[:, :-1, :6]
    delta_sigma = torch.cat(
        [strain_previous[:, :1, :], strain_current - strain_previous], dim=1
    )

    # Compute stress dot product change
    stress_dot_change = torch.sum(y_pred * delta_sigma, dim=[1, 2])
    t = 1.0
    relu_term = nn.functional.relu(-t * stress_dot_change) # TODO: this penalizes negative work increment such as during unloading. I think this is an incorrect approach
    term3 = torch.mean(relu_term)

    # Print debug information
    # print("term1:", term1.item(), "term2:", term2.item(), "term3:", term3.item(), "sum:", (term1 + term2 + term3).item())

    return term1 + term3  # Return the final loss (excluding term2)


# Train the model for the loss function loss_fn
def train(model, X_train, y_train, extra_args_list_train, batch_size, epochs, learning_rate, loss_fn, loss_fn_extra_args_list):
    # Signature of loss function must be: fn(model, X, y, extra_args, loss_fn_extra_args_list)
    if (batch_size > X_train.shape[0]):
        print("WARNING: batch_size larger than size of data")
        print(f"WARNING: original batch_size ({batch_size}) decreased to {X_train.shape[0]}")
        batch_size = X_train.shape[0]
    model.train()

    # Create DataLoader and optimizer
    train_dataset = TensorDataset(X_train, y_train) if (extra_args_list_train == None) else TensorDataset(X_train, y_train, *extra_args_list_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss_epoch = 0.0
        for X_batch, y_batch, *extra_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model, X_batch, y_batch, extra_batch if len(extra_batch) > 0 else None, loss_fn_extra_args_list)
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
        print(f'Epoch {epoch + 1}/{epochs} - Total Loss: {total_loss_epoch}')
    return total_loss_epoch # Return loss at last training epoch


# Optimize hyperparameters of the model
def optimize_hyperparameters(modelClass, model_input_size, model_hyperparams, training_hyperparams, device, X_train, y_train, extra_args_list_train, epochs, n_trials, loss_fn, loss_fn_extra_args_list):
    # model_hyperparams = list of {dic of trial.suggest_type() arguments} for batch_size and learning_rate
    # model_hyperparams =  list of ('type', {dic of trial.suggest_type() arguments}) for network parameters
    # Assumes modelClass constructor is of the form modelClass(model_input_size, hyperparameter1, hyperparameter2...)

    # Run Optuna Optimization
    study = optuna.create_study(direction="minimize")  # Minimize the loss

    # Define objective function for hyperparameters optimization in Optuna
    def objective_Optuna(trial):
        # Sample hyperparameters
        batch_size = trial.suggest_int(**training_hyperparams[0])
        learning_rate = trial.suggest_float(**training_hyperparams[1])

        # Model hyperparameters
        model_args = [model_input_size]
        for (datatype, hyperparam) in model_hyperparams:
            if (datatype == 'int'):
                model_args += [trial.suggest_int(**hyperparam)]
            elif (datatype == 'float'):
                model_args += [trial.suggest_float(**hyperparam)]

        model = modelClass(*tuple(model_args)).to(device)
        # Return final loss for Optuna to minimize
        return train(model, X_train, y_train, extra_args_list_train, batch_size, epochs, learning_rate, loss_fn, loss_fn_extra_args_list)

    study.optimize(objective_Optuna, n_trials=n_trials)
    return study.best_params

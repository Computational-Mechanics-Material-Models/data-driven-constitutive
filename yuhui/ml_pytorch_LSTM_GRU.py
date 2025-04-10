#!/usr/bin/env python

# PyTorch Script for Long Short-Term Memory (LSTM) networks
# and Gated Recursive Units (GRU) networks to model quasi-brittle materials

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


# Check available GPUs
physical_devices = torch.cuda.device_count()
print(f"Available GPUs: {physical_devices}")


# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Choose the desired GPU index
gpu_id = 4
if gpu_id < physical_devices:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)  # Set the current device
    print(f"Binding to GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    device = torch.device("cpu")
    print(f"GPU {gpu_id} not available, using CPU instead.")
# Example: Move a tensor to the selected GPU
tensor_example = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(tensor_example.device)  # Should print "cuda:4" if GPU 4 is available


# Read in and process file containing loading history for multiple tests
# step, strain11, strain22, strain33, strain12, strain13, strain23, stress11, stress22, stress33, stress12, stress13, stress23, index, size, material
# 1000 steps = [0 ; 999] vertically stacked for a given test, one step per row
# 100 blocks (index = [0 ; 99]) of 1000 steps vertically stacked. What is the index representing ? One index per test ?
# material = 0 for all tests
# size = 30 for all tests
# TODO: I have no idea what units stress, strain, size are!
# TODO: Is there is any normalization?
# TODO: Pandas uses its own `index`, so using a variable named `index` might be confusing. Consider changing name
file_paths = ["averaged_size_30_strain22.csv",]
df_list = [pd.read_csv(file) for file in file_paths]

# Make index unique for all files in in case multiple files with similar format
# i.e., index start at zero are in `file_paths`
index_offset = 0
for df in df_list:
    df['index'] = df['index'] + index_offset
    index_offset += len(df['index'].unique())

# Combine clean and export consolidated data in files
df_combined = pd.concat(df_list, ignore_index=False)
df_combined.dropna(inplace=True)
df_combined = df_combined.astype(np.float64) # TODO: float 32 precise enough for ML applications?
print(df_combined.head(1001)) # TODO: Do not use hardcoded value !!!
df_combined.to_csv("combined_dataset.csv", index=False)


# TODO: Ask Yuhui: files not provided
# TODO: The `strain_generation` script does not create files in .txt format or with angle1 name
# TODO: No idea how these files should be obtained...

angle1 = np.genfromtxt("angle1.txt", delimiter=',')
angle2 = np.genfromtxt("angle2.txt", delimiter=',')
angle3 = np.genfromtxt("angle3.txt", delimiter=',')


# In[ ]:

# TODO consider moving function definitions at start of file
def generateRmatrix(angle1, angle2, angle3):
    R1 = np.array([[np.cos(angle1), -np.sin(angle1), 0],[np.sin(angle1), np.cos(angle1), 0],[0, 0, 1]])
    print(R1.shape)
    R2 = np.array([[np.cos(angle2), 0, np.sin(angle2)], [0,1,0], [-np.sin(angle2), 0, np.cos(angle2)]])
    R3 = np.array([[1, 0, 0], [0, np.cos(angle3), -np.sin(angle3)], [0, np.sin(angle3), np.cos(angle3)]])
    R = np.matmul(np.matmul(R1, R2), R3)
    return R

R=generateRmatrix(angle1[0], angle2[0], angle3[0])
print(R)


# In[ ]:
    
# TODO JBC: Looks like what was in this Jupyter block and the next one could be
# manually chosen by the user to use either / or techniques to extract X and y
# from the data. To keep them and have them work in a regular script,
# I refactored them into functions
def extract_input_and_output(df_combined,
                             sequence_length = 1000,
                             input_n_features = 6,
                             output_n_features = 6,
                             input_columns = ["strain11", "strain22", "strain33", "strain12", "strain13", "strain23"],
                             output_columns = ["stress11", "stress22", "stress33", "stress12", "stress13", "stress23"]):
    # 重新计算符合条件的样本数
    valid_indices = df_combined['index'].unique()
    count = len(valid_indices)   # 每个 index 有 3 组数据

    X = np.zeros((count, sequence_length, input_n_features))
    y = np.zeros((count, sequence_length, output_n_features))

    count = 0
    for i in valid_indices:
        df1 = df_combined[df_combined['index'] == i]
        df1 = df1.sort_values(by="step") # In case steps are not ordered (TODO: can be disordered by previous line? or always in order and this unnecessary?)

        # 按 step 递增分成三组，每组取 1000 行
        # TODO: I don't understand what is happening here
        # Unless there can be more than 1000 steps for a given index, df1 should always be size 1000
        # and the code below is pointless as j=0 will always return. What am I missing?
        for j in range(3):
            subset = df1.iloc[j * sequence_length : (j + 1) * sequence_length]
            if len(subset) == sequence_length:
                X[count] = subset[input_columns].to_numpy()
                y[count] = subset[output_columns].to_numpy()
                count += 1
    return X, y

X, y = extract_input_and_output(df_combined)
print(X.shape, y.shape)

# TODO JBC: Looks like what was in this Jupyter block and the previous one could be
# manually chosen by the user to use either / or techniques to extract X and y
# from the data. To keep them and have them work in a regular script,
# I refactored them into functions
def extract_and_normalize_input_and_output(df_combined,
                                           sequence_length = 1000,
                                           input_n_features = 6,
                                           output_n_features = 6,
                                           input_columns = ["strain11", "strain22", "strain33", "strain12", "strain13", "strain23"],
                                           output_columns = ["stress11", "stress22", "stress33", "stress12", "stress13", "stress23"]):
    # 重新计算符合条件的样本数
    valid_indices = df_combined['index'].unique()
    count = len(valid_indices)   # 每个 index 有 3 组数据

    X = np.zeros((count, sequence_length, input_n_features))
    y = np.zeros((count, sequence_length, output_n_features))

    count = 0
    for i in valid_indices:
        df1 = df_combined[df_combined['index'] == i]
        df1 = df1.sort_values(by="step") # In case steps are not ordered (TODO: can be disordered by previous line? or always in order and this unnecessary?)

        # 按 step 递增分成三组，每组取 1000 行
        # TODO: I don't understand what is happening here
        # Unless there can be more than 1000 steps for a given index, df1 should always be size 1000
        # and the code below is pointless as j=0 will always return. What am I missing?
        for j in range(3):
            subset = df1.iloc[j * sequence_length : (j + 1) * sequence_length]
            if len(subset) == sequence_length:
                # Standardize inputs to unit variance
                # TODO: is the mean of the input zero ? If not, doing this changes the mean !
                std_devs_in = subset[input_columns].std().to_numpy()
                std_devs_out = subset[output_columns].std().to_numpy()
                # Avoid division by zero
                std_devs_in[std_devs_in <= 0] = 1e-6
                std_devs_out[std_devs_out <= 0] = 1e-6
                X[count] = subset[input_columns].to_numpy() / std_devs_in
                y[count] = subset[output_columns].to_numpy() / std_devs_out

                # TODO: DO NOT SHARE UNDOCUMENTED COMMENTED CODE !!!!!!!!!!!!!
                # TODO: SHOULD THIS BE USED? WHEN? HOW?
                # NO ONE KNOWS! AND THOSE WHO DO WILL EVENTUALLY FORGET OR CHANGE JOB !

                        #  # Compute normalization parameters
                        #  X_min = subset[input_columns].min().to_numpy()  # Minimum values for each feature
                        #  X_max = subset[input_columns].max().to_numpy()  # Maximum values for each feature

                        #  X_m = (X_min + X_max) / 2  # Mean of min and max
                        #  X_s = (X_max - X_min) / 2  # Scaling factor

                        # # Normalize X using the given formula
                        #  X[count] = (subset[input_columns].to_numpy() - X_m) / X_s

                        # # Compute normalization parameters for y
                        #  y_min = subset[output_columns].min().to_numpy()
                        #  y_max = subset[output_columns].max().to_numpy()

                        #  y_m = (y_min + y_max) / 2
                        #  y_s = (y_max - y_min) / 2

                        #  # Normalize y using the given formula
                        #  y[count] = (subset[output_columns].to_numpy() - y_m) / y_s
                        #  # X[count] = subset[input_columns].to_numpy()
                        #  # y[count] = subset[output_columns].to_numpy()

                # Normalize input data in range [0, 1]
                # TODO: Normalizing between [0, 1] changes the variance again
                # so the standardization we did prior might be better done at the end
                # and normalization with a zero mean, e.g., between [-1, 1] might be preferred
                X_min = subset[input_columns].min().to_numpy()
                X_max = subset[input_columns].max().to_numpy()
                X[count] = (subset[input_columns].to_numpy() - X_min) / (X_max - X_min)
                # Normalize output data in range [0, 1]
                # TODO: Normalizing between [0, 1] changes the variance again
                # so the standardization we did prior might be better done at the end
                # and normalization with a zero mean, e.g., between [-1, 1] might be preferred
                y_min = subset[output_columns].min().to_numpy()
                y_max = subset[output_columns].max().to_numpy()
                y[count] = (subset[output_columns].to_numpy() - y_min) / (y_max - y_min)
                count += 1
    return X, y

X, y = extract_and_normalize_input_and_output(df_combined)
X.shape, y.shape


# In[ ]:


import torch
import torch.nn.functional as F

# Define custom loss function in PyTorch
def make_custom_loss_batch(model, X_batch):
    def custom_loss(y_pred, y_true):
        # Convert R to a PyTorch tensor
        R_tensor = torch.tensor(R, dtype=torch.float32, device=y_pred.device) # TODO: DO NOT USE GLOBALS LIKE `R` !
        strain = X_batch.to(dtype=torch.float32)

        # Extract the diagonal strain components (first three features)
        diagonal_strain = strain[:, :, :3]  # Shape: (batch_size, 1000, 3)

        # Compute Term 1: MSE loss between predictions and ground truth
        term1 = torch.mean(torch.sum((y_pred - y_true) ** 2, dim=[1, 2]))

        # Compute Term 2: Rotation-based transformation
        rotated_strain = torch.matmul(diagonal_strain.view(-1, 3), R_tensor)
        rotated_strain = rotated_strain.view(diagonal_strain.shape)  # Reshape to original shape

        # Compute R^{-1} (inverse of R)
        R_tensor_inv = torch.linalg.inv(R_tensor)  
        transformed_strain = torch.matmul(rotated_strain.view(-1, 3), R_tensor_inv)
        transformed_strain = transformed_strain.view(rotated_strain.shape)

        # Pad transformed strain to match the input shape
        transformed_strain_padded = torch.cat(
            [transformed_strain, torch.zeros_like(strain[:, :, 3:])], dim=-1
        )

        # Predict stress using the model
        predicted_transformed_stress = model(transformed_strain_padded)  

        # Apply rotation matrix to stress
        rotated_stress = torch.matmul(y_pred[:, :, :3].reshape(-1, 3), R_tensor)
        rotated_stress = rotated_stress.view(y_pred[:, :, :3].shape)

        # Compute Term 2 difference
        difference = predicted_transformed_stress[:, :, :3] - rotated_stress
        term2 = torch.mean(torch.sum(difference ** 2, dim=[1, 2]))

        # Compute Term 3: Delta stress change
        strain_current = strain[:, 1:, :6]  
        strain_previous = strain[:, :-1, :6]  
        delta_sigma = torch.cat(
            [strain_previous[:, :1, :], strain_current - strain_previous], dim=1
        )

        # Compute stress dot product change
        stress_dot_change = torch.sum(y_pred * delta_sigma, dim=[1, 2])
        t = 1.0
        relu_term = F.relu(-t * stress_dot_change)
        term3 = torch.mean(relu_term)

        # Print debug information
        print("term1:", term1.item(), "term2:", term2.item(), "term3:", term3.item(), "sum:", (term1 + term2 + term3).item())

        return term1 + term3  # Return the final loss (excluding term2)

    return custom_loss

# In[ ]:


# Convert dataset to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout1, dropout2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_dim2, 6)  # Output layer for regression

    def forward(self, x):
        print(f"Input shape to LSTM: {x.shape}")  # Debugging
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x) # Select last timestep for prediction
        print(f"Output shape from LSTM: {x.shape}")  # Debugging
        return x

# Define Optuna Objective Function
def objective(trial):
    # Sample hyperparameters
    batch_size = trial.suggest_int("batch_size", 16,64, step=8)
    lstm_units_1 = trial.suggest_int("lstm_units_1", 32, 128, step=16)
    lstm_units_2 = trial.suggest_int("lstm_units_2", 16, 64, step=16)
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Create DataLoader (Fix 1: drop_last=True to ensure equal batch sizes)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model
    model = LSTMModel(input_dim=6, hidden_dim1=lstm_units_1, hidden_dim2=lstm_units_2,
                      dropout1=dropout_1, dropout2=dropout_2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    # criterion = nn.MSELoss()  # Mean Squared Error for regression task

    # # Debugging: Check first batch shapes
    # for X_batch, y_batch in train_loader:
    #     print(f"Batch X shape: {X_batch.shape}")  # Expected: (batch_size, sequence_length, num_features)
    #     print(f"Batch y shape: {y_batch.shape}")  # Expected: (batch_size, 6)
    #     break

    # Training loop
    epochs = 300
    for epoch in range(epochs):
        total_loss_epoch = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            # loss = criterion(y_pred, y_batch)
            # Get the loss function dynamically for this batch
            loss_fn = make_custom_loss_batch(model, X_batch)  
            loss = loss_fn(y_pred, y_batch)  
            
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
        print(f'Epoch {epoch + 1}, Total Loss: {total_loss_epoch}')
    
    return total_loss_epoch  # Return final loss for Optuna to minimize

# Run Optuna Optimization
study = optuna.create_study(direction="minimize")  # Minimize the loss
study.optimize(objective, n_trials=10)  # Reduce trials for debugging

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)




# In[ ]:

        
def train_model(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss_epoch = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)

            # Get the loss function dynamically for this batch
            loss_fn = make_custom_loss_batch(model, X_batch)  
            loss = loss_fn(y_pred, y_batch)  

            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
        
        print(f'Epoch {epoch + 1}/{epochs} - Total Loss: {total_loss_epoch}')




# Hyperparameters
batch_size = 56
lstm_units_1 = 80
lstm_units_2 = 64
dropout_1 = 0.1
dropout_2 = 0.3
learning_rate = 0.005
epochs = 1000  # Adjust for testing

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize Model, Optimizer, Loss Function
model = LSTMModel(input_dim=6, hidden_dim1=lstm_units_1, hidden_dim2=lstm_units_2,
                  dropout1=dropout_1, dropout2=dropout_2).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Model
train_model(model, train_loader, optimizer, epochs)


# In[ ]:


# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from sklearn.metrics import r2_score

# # Ensure model is in evaluation mode
# model.eval()

# # Create a directory to save plots
# save_dir = 'test_set_plots_3'
# os.makedirs(save_dir, exist_ok=True)

# # print(X.shape, y.shape, X_test.shape, y_test.shape, X_train.shape, y_train.shape)


# # Ensure X_test is a PyTorch tensor and move to the correct device
# X_test = X_test.to(next(model.parameters()).device)

# # print(X.shape, y.shape, X_test.shape, y_test.shape, X_train.shape, y_train.shape)


# # Make predictions for the test set (disable gradients)
# with torch.no_grad():
#     predictions = model(X_test)  # Forward pass

# # Convert predictions and tensors back to NumPy
# predictions = predictions.cpu().numpy()
# X_test_np = X_test.cpu().numpy()
# y_test_np = y_test.cpu().numpy()

# # Number of test samples
# num_tests = X_test.shape[0]

# # print(X.shape, y.shape, X_test.shape, y_test.shape, X_train.shape, y_train.shape)

# # Loop over each test sample to plot
# for i in range(num_tests):
#     # Extract strain_11 (component 0 of strain tensor)
#     strain_11 = X_test_np[i, :, 1]  # Strain in the first direction (epsilon_11)
#     # print(strain_11)
    
#     # Extract true stress_11 (component 0 of stress tensor)
#     true_stress_11 = y_test_np[i, :, 1]  # True stress in the first direction (sigma_11)
#     # Extract predicted stress_11 (component 0 of predicted stress tensor)
#     predicted_stress_11 = predictions[i, :, 1]  # Predicted stress in the first direction (sigma_11)

#     # Compute R² score
#     r2 = r2_score(true_stress_11, predicted_stress_11)

#     # Plot true stress_11 and predicted stress_11 against strain_11
#     plt.figure(figsize=(8, 6))
#     plt.plot(strain_11, true_stress_11, label='True Stress_11', color='blue', marker='o')
#     plt.plot(strain_11, predicted_stress_11, label='Predicted Stress_11', color='red', linestyle='--')

#     # Labeling the plot
#     plt.title(f'Test Sample {i+1}: Stress_11 vs Strain_11 (R2 = {r2:.4f})')
#     plt.xlabel('Strain_11 (epsilon_11)')
#     plt.ylabel('Stress_11 (sigma_11)')
#     plt.legend()

#     # Show plot
#     plt.show()

#     # Save plot as an image file
#     plt.savefig(f'{save_dir}/plot_example_{i}.png')

#     # Close the figure to free memory
#     plt.close()


# In[ ]:


# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from sklearn.metrics import r2_score

# # Ensure model is in evaluation mode
# model.eval()

# # Create a directory to save plots
# save_dir = 'test_set_plots_3'
# os.makedirs(save_dir, exist_ok=True)

# # Ensure X_test is a PyTorch tensor and move to the correct device
# X_train = X_train.to(next(model.parameters()).device)
# print(X_train.shape)
# # Make predictions for the test set (disable gradients)
# with torch.no_grad():
#     predictions = model(X_train)  # Forward pass

# # Convert predictions and tensors back to NumPy
# predictions = predictions.cpu().numpy()
# X_train_np = X_train.cpu().numpy()
# y_train_np = y_train.cpu().numpy()

# # Number of test samples
# num_train = X_train.shape[0]
# # print(num_tests, X_train)
# # Loop over each test sample to plot  
# for i in range(num_train):
#     # Extract strain_11 (component 0 of strain tensor)
#     strain_11 = X_train_np[i, :, 1]  # Strain in the first direction (epsilon_11)
#     # print(len( strain_11))
#     # Extract true stress_11 (component 0 of stress tensor)
#     true_stress_11 = y_train_np[i, :, 1]  # True stress in the first direction (sigma_11)

#     # Extract predicted stress_11 (component 0 of predicted stress tensor)
#     predicted_stress_11 = predictions[i, :, 1]  # Predicted stress in the first direction (sigma_11)

#     # Compute R² score
#     r2 = r2_score(true_stress_11, predicted_stress_11)

#     # Plot true stress_11 and predicted stress_11 against strain_11
#     plt.figure(figsize=(8, 6))
#     plt.plot(strain_11, true_stress_11, label='True Stress_11', color='blue', marker='o')
#     plt.plot(strain_11, predicted_stress_11, label='Predicted Stress_11', color='red', linestyle='--')

#     # Labeling the plot
#     plt.title(f'Train Sample {i+1}: Stress_11 vs Strain_11 (R2 = {r2:.4f})')
#     plt.xlabel('Strain_11 (epsilon_11)')
#     plt.ylabel('Stress_11 (sigma_11)')
#     plt.legend()

#     # Show plot
#     plt.show()

#     # Save plot as an image file
#     plt.savefig(f'{save_dir}/plot_example_{i}.png')

#     # Close the figure to free memory
#     plt.close()


# In[ ]:
    
###GRU

R=generateRmatrix(angle1[0], angle2[0], angle3[0])
print(R)




# In[ ]:

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout1, dropout2):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_dim2, 6)  # Output layer for regression

    def forward(self, x):
        print(f"Input shape to GRU: {x.shape}")  # Debugging
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = self.fc(x) 
        print(f"Output shape from GRU: {x.shape}")  # Debugging
        return x

# Define Optuna Objective Function
def objective(trial):
    # Sample hyperparameters
    batch_size = trial.suggest_int("batch_size", 16, 64, step=8)
    gru_units_1 = trial.suggest_int("gru_units_1", 32, 128, step=16)
    gru_units_2 = trial.suggest_int("gru_units_2", 16, 64, step=16)
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize GRU model
    model = GRUModel(input_dim=6, hidden_dim1=gru_units_1, hidden_dim2=gru_units_2,
                     dropout1=dropout_1, dropout2=dropout_2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    epochs = 300
    for epoch in range(epochs):
        total_loss_epoch = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            # Get the loss function dynamically for this batch
            loss_fn = make_custom_loss_batch(model, X_batch)  
            loss = loss_fn(y_pred, y_batch)  
            
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
        print(f'Epoch {epoch + 1}, Total Loss: {total_loss_epoch}')
    
    return total_loss_epoch  # Return final loss for Optuna to minimize

# Run Optuna Optimization
study = optuna.create_study(direction="minimize")  # Minimize the loss
study.optimize(objective, n_trials=10)  # Reduce trials for debugging

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)


# In[ ]:

# Hyperparameters
batch_size = 56
lstm_units_1 = 112
lstm_units_2 = 16
dropout_1 = 0.1
dropout_2 = 0.3
learning_rate = 0.0085
epochs = 1000  # Adjust for testing

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize Model, Optimizer, Loss Function
model = LSTMModel(input_dim=6, hidden_dim1=lstm_units_1, hidden_dim2=lstm_units_2,
                  dropout1=dropout_1, dropout2=dropout_2).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Model
train_model(model, train_loader, optimizer, epochs)


# In[ ]:


# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from sklearn.metrics import r2_score

# # Ensure model is in evaluation mode
# model.eval()

# # Create a directory to save plots
# save_dir = 'test_set_plots_3'
# os.makedirs(save_dir, exist_ok=True)

# # print(X.shape, y.shape, X_test.shape, y_test.shape, X_train.shape, y_train.shape)


# # Ensure X_test is a PyTorch tensor and move to the correct device
# X_test = X_test.to(next(model.parameters()).device)

# # print(X.shape, y.shape, X_test.shape, y_test.shape, X_train.shape, y_train.shape)


# # Make predictions for the test set (disable gradients)
# with torch.no_grad():
#     predictions = model(X_test)  # Forward pass

# # Convert predictions and tensors back to NumPy
# predictions = predictions.cpu().numpy()
# X_test_np = X_test.cpu().numpy()
# y_test_np = y_test.cpu().numpy()

# # Number of test samples
# num_tests = X_test.shape[0]

# # print(X.shape, y.shape, X_test.shape, y_test.shape, X_train.shape, y_train.shape)

# # Loop over each test sample to plot
# for i in range(num_tests):
#     # Extract strain_11 (component 0 of strain tensor)
#     strain_11 = X_test_np[i, :, 1]  # Strain in the first direction (epsilon_11)
#     # print(strain_11)
    
#     # Extract true stress_11 (component 0 of stress tensor)
#     true_stress_11 = y_test_np[i, :, 1]  # True stress in the first direction (sigma_11)
#     # Extract predicted stress_11 (component 0 of predicted stress tensor)
#     predicted_stress_11 = predictions[i, :, 1]  # Predicted stress in the first direction (sigma_11)

#     # Compute R² score
#     r2 = r2_score(true_stress_11, predicted_stress_11)

#     # Plot true stress_11 and predicted stress_11 against strain_11
#     plt.figure(figsize=(8, 6))
#     plt.plot(strain_11, true_stress_11, label='True Stress_11', color='blue', marker='o')
#     plt.plot(strain_11, predicted_stress_11, label='Predicted Stress_11', color='red', linestyle='--')

#     # Labeling the plot
#     plt.title(f'Test Sample {i+1}: Stress_11 vs Strain_11 (R2 = {r2:.4f})')
#     plt.xlabel('Strain_11 (epsilon_11)')
#     plt.ylabel('Stress_11 (sigma_11)')
#     plt.legend()

#     # Show plot
#     plt.show()

#     # Save plot as an image file
#     plt.savefig(f'{save_dir}/plot_example_{i}.png')

#     # Close the figure to free memory
#     plt.close()



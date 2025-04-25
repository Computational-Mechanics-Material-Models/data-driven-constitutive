#!/usr/bin/env python

# PyTorch demo using Logarzo RNN architecture with LSTM and GRU cells to model quasi-brittle materials
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Add models and training routines from external files
# TODO: this is temporary. we should create a python package to do that cleanly
import sys, os
sys.path.append(os.path.abspath('..')) # TODO: hardcoded path only works if working dir is `demo`
import utils_yuhuilyu_data
from models import logarzo_lstm as LSTMModel
from models import logarzo_gru as GRUModel
from training import loss_yuhuilyu
from training import train
from training import optimize_hyperparameters

def main():
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

    # Import and normalize data
    normalization = 'scale'
    normalization_interval = (1.0 / 6e-4, 1.0 / 4e6) # In the data file below: |strain| < 6e-4 ; |stress| < 4e6 Pa
    X, y, R = utils_yuhuilyu_data.get_data(["averaged_size_30_strain22.csv",], normalization_style=normalization, normalization_values=normalization_interval) # Hardcoded path for now, file must be in `demo` dir
    study_ndx = 300 # To restrict the time sequence of studied data
    X = X[:, :study_ndx,:] # Strain history [batch_size, sequence_length, features]
    y = y[:, :study_ndx,:] # Stress history [batch_size, sequence_length, features]
    # Convert dataset to PyTorch tensors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # ---------- #
    # LSTM model #
    # ---------- #
    training_hyperparams_lstm = [
        {'name':"batch_size", 'low':16, 'high':64, 'step':8},
        {'name':"learning_rate", 'low':1e-5, 'high':1e-2, 'log':True}
        ]
    model_hyperparams_lstm = [
        ('int', {'name':"hidden_dim1", 'low':32, 'high':128, 'step':16}),
        ('int', {'name':"hidden_dim2", 'low':16, 'high':64, 'step':16}),
        ('float', {'name':"dropout1", 'low':0.1, 'high':0.5, 'step':0.1}),
        ('float', {'name':"dropout2", 'low':0.1, 'high':0.5, 'step':0.1})
        ]
    input_dim_lstm = 6

    # --------- #
    # GRU model #
    # --------- #
    training_hyperparams_gru = [
        {'name':"batch_size", 'low':16, 'high':64, 'step':8},
        {'name':"learning_rate", 'low':1e-5, 'high':1e-2, 'log':True}
        ]
    model_hyperparams_gru = [
        ('int', {'name':"hidden_dim1", 'low':32, 'high':128, 'step':16}),
        ('int', {'name':"hidden_dim2", 'low':16, 'high':64, 'step':16}),
        ('float', {'name':"dropout1", 'low':0.1, 'high':0.5, 'step':0.1}),
        ('float', {'name':"dropout2", 'low':0.1, 'high':0.5, 'step':0.1})
        ]
    input_dim_gru = 6


    # Train models
    models = []
    model_names = ("LSTM", "GRU")
    models_forward_extra_args = None # Models do not take extra arguments in forward() method

    epochs_hyperparams = 300 # fewer amount for hyperparameters training
    n_trials = 10 # Reduce trials for debugging
    epochs_training = 1000  # Adjust for testing

    for modelClass, training_hyperparams, model_hyperparams, input_dim, name in zip((LSTMModel, GRUModel),
                                                                                    (training_hyperparams_lstm, training_hyperparams_gru),
                                                                                    (model_hyperparams_lstm, model_hyperparams_gru),
                                                                                    (input_dim_lstm, input_dim_gru),
                                                                                    model_names):
        print(name)
        # Training hyperparameters
        hyperparams = optimize_hyperparameters(modelClass, input_dim, model_hyperparams, training_hyperparams, device, X_train, y_train, models_forward_extra_args, epochs_hyperparams, n_trials, loss_yuhuilyu, R)
        print(f"Best hyperparameters {name}:", hyperparams)

        # Train model using optimized Hyperparameters
        batch_size = hyperparams['batch_size']
        learning_rate = hyperparams['learning_rate']
        model_hyperparams = {key:hyperparams[key] for key in hyperparams if (key != 'batch_size' and key != 'learning_rate')}

        model = modelClass(input_dim=input_dim, **model_hyperparams).to(device)
        train(model, X_train, y_train, models_forward_extra_args, batch_size, epochs_training, learning_rate, loss_yuhuilyu, R)
        models += [model]

        # TODO: store trained weights to file for use in constitutive model
        # print(model.lstm1.weight_ih_l0.detach().numpy())
        # print(model.lstm1.bias_ih_l0.detach().numpy())
        # print(model.lstm1.weight_hh_l0.detach().numpy())
        # print(model.lstm1.bias_hh_l0.detach().numpy())
        # print(model.lstm1.all_weights)
        # print(model.fc.weight.detach().numpy())
        # print(model.fc.bias.detach().numpy())


    # Test models
    predictions = []
    for model in models:
        # Evaluation mode to remove effect of dropout layers. Disable gradients
        model.eval()
        with torch.no_grad():
            predictions += [model(X_test)] # No extra forward arguments


    # Plot results and comparison between LSTM and GRU
    for i, (strain, stress_data, stress_LSTM, stress_GRU) in enumerate(zip(X_test.cpu()[:,:,0], y_test.cpu()[:,:,0], predictions[0].cpu()[:,:,0], predictions[1].cpu()[:,:,0])):
        r2_lstm = r2_score(stress_data, stress_LSTM) # Compute R^2 score
        r2_gru = r2_score(stress_data, stress_GRU) # Compute R^2 score

        # Plot true stress_11 and predicted stress_11 against strain_11
        plt.figure(figsize=(8, 6))
        plt.plot(strain, stress_data, label='Data', color='grey', marker='o', linestyle='none')
        plt.plot(strain, stress_LSTM, label='LSTM', color='red', linestyle='--')
        plt.plot(strain, stress_GRU, label='GRU', color='blue', linestyle='--')
        plt.title(f'Test Sample {i+1}: (R2_LSTM = {r2_lstm:.4f} , R2_GRU = {r2_gru:.4f})')
        plt.xlabel('e11')
        plt.ylabel('s11')
        plt.legend()
        plt.show()
        plt.savefig(f'plot_example_{i}.png')
        plt.close()

if __name__ == "__main__":
    main()

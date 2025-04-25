#!/usr/bin/env python

# PyTorch demo using simple linear RNN to model quasi-brittle materials
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Add models and training routines from external files
# TODO: this is temporary. we should create a python package to do that cleanly
import sys, os
sys.path.append(os.path.abspath('..')) # TODO: hardcoded path only works if working dir is dir containing this script file
import utils_yuhuilyu_data
from models import rnn_linear
from training import loss_yuhuilyu
from training import train

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

    study_ndx = 300 # To restrict the time sequence of studied data
    strain = strain[:, :study_ndx,:] # Strain history [batch_size, sequence_length, features]
    stress = stress[:, :study_ndx,:] # Stress history [batch_size, sequence_length, features]
    # Import and normalize data
    normalization = '[0,1]'
    strain, stress, R = utils_yuhuilyu_data.get_data(["averaged_size_30_strain22.csv",], normalization=normalization) # Hardcoded path for now, file must be in `demo` dir
    # Convert dataset to PyTorch tensors
    strain_train, strain_test, stress_train, stress_test = train_test_split(strain, stress, test_size=0.25, random_state=42)
    strain_train = torch.tensor(strain_train, dtype=torch.float32).to(device)
    stress_train = torch.tensor(stress_train, dtype=torch.float32).to(device)
    strain_test = torch.tensor(strain_test, dtype=torch.float32).to(device)
    stress_test = torch.tensor(stress_test, dtype=torch.float32).to(device)


    # ----------------- #
    # Simple linear RNN #
    # ----------------- #
    num_hidden = 2 # 2 hidden layers
    size_hidden = 100 # 100 neurons per layer
    activation = 'relu' # ReLU activation
    training_style = 'direct' # Direct training (Xu et al. 2021), i.e., use previous stress from data rather than recursive estimate from model
    model = rnn_linear(num_hidden, size_hidden, activation, training_style)

    # Train model
    batch_size = 64
    epochs = 1000
    learning_rate = 0.005
    model_forward_extra_args_list = (stress_train,) # Stress history passed to forward() for direct training
    train(model, strain_train, stress_train, model_forward_extra_args_list, batch_size, epochs, learning_rate, loss_yuhuilyu, R)
    
    # Test model
    model.eval() # In evaluation mode, model computes stress recursively
    with torch.no_grad():
        prediction = model(strain_test) # No stress history provided for recursive evaluation

    # This performs VERY POORLY. Massive error accumulates. Not a good idea
    # This is what Xu et al. 2021 call "dircet training". It seems like a good idea, why is it so bad? Maybe because I used direct data instead of increment
    
    # Plot results and comparison for ff linear
    for i, (eps, sig, sig_pred) in enumerate(zip(strain_test.cpu()[:,:study_ndx,0], stress_test.cpu()[:,:study_ndx,0], prediction.cpu()[:,:study_ndx,0])):
        r2 = r2_score(sig, sig_pred) # Compute R^2 score

        # Plot true stress_11 and predicted stress_11 against strain_11
        plt.figure(figsize=(8, 6))
        plt.plot(eps, sig, label='Data', color='grey', marker='o', linestyle='none')
        plt.plot(eps, sig_pred, label='linear rnn, direct training', color='red', linestyle='--')
        plt.title(f'Test Sample {i+1}: (R2_ {r2:.4f})')
        plt.xlabel('e11')
        plt.ylabel('s11')
        plt.legend()
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
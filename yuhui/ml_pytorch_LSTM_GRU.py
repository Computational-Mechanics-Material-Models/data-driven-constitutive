#!/usr/bin/env python

# PyTorch Script for Long Short-Term Memory (LSTM) networks
# and Gated Recursive Units (GRU) networks to model quasi-brittle materials

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import  qmc


# TODO: Is there a better design to test objectivity?
# It makes it really clunky to pass it to the function and do the rotations there!
# TODO: if we don't need to memorize what this matrix should be, we could compute
# it on the fly inside the loss function and avoid all this...
def generateRmatrix(angle1, angle2, angle3):
    R1 = np.array([[np.cos(angle1), -np.sin(angle1), 0],[np.sin(angle1), np.cos(angle1), 0],[0, 0, 1]])
    print(R1.shape)
    R2 = np.array([[np.cos(angle2), 0, np.sin(angle2)], [0,1,0], [-np.sin(angle2), 0, np.cos(angle2)]])
    R3 = np.array([[1, 0, 0], [0, np.cos(angle3), -np.sin(angle3)], [0, np.sin(angle3), np.cos(angle3)]])
    R = np.matmul(np.matmul(R1, R2), R3)
    return R






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


def main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    # Add models and training routines from external files
    # TODO: this is temporary. we should create a python package to do that cleanly
    import sys, os
    sys.path.append(os.path.abspath('..')) # TODO: hardcoded path only works if working dir is dir containing this script file
    from models import logarzo_lstm as LSTMModel
    from models import logarzo_gru as GRUModel
    from training import loss_yuhuilyu
    from training import train
    from training import optimize_hyperparameters

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


    # Generate random orientation to weakly enforce objectivity through the
    # objective function.
    ndim_angle = 3
    ndata_angle = 300
    sampler = qmc.LatinHypercube(d = ndim_angle)
    sample = sampler.random(n = ndata_angle)
    angle1 = sample[:,0] * 2 * np.pi
    angle2 = sample[:,1] * 2 * np.pi
    angle3 = sample[:,2] * 2 * np.pi
    # angle1 = np.genfromtxt("angle1.txt", delimiter=',')
    # angle2 = np.genfromtxt("angle2.txt", delimiter=',')
    # angle3 = np.genfromtxt("angle3.txt", delimiter=',')
    
    # TODO: right now the script only uses a single matrix
    # What is the expected design to test for many random configurations?
    R=generateRmatrix(angle1[0], angle2[0], angle3[0])
    print(R)

    # TODO: figure out which one I am supposed to use and delete the other
    X, y = extract_input_and_output(df_combined)
    print(X.shape, y.shape)
    X, y = extract_and_normalize_input_and_output(df_combined)
    X.shape, y.shape


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
        hyperparams = optimize_hyperparameters(modelClass, input_dim, model_hyperparams, training_hyperparams, device, X_train, y_train, epochs_hyperparams, n_trials, models_forward_extra_args, loss_yuhuilyu, R)
        print(f"Best hyperparameters {name}:", hyperparams)

        # Train model using optimized Hyperparameters
        batch_size = hyperparams['batch_size']
        learning_rate = hyperparams['learning_rate']
        model_hyperparams = {key:hyperparams[key] for key in hyperparams if (key != 'batch_size' and key != 'learning_rate')}

        model = modelClass(input_dim=input_dim, **model_hyperparams).to(device)
        train(model, X_train, y_train, batch_size, epochs_training, learning_rate, models_forward_extra_args, loss_yuhuilyu, R)
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
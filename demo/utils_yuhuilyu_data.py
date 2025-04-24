#!/usr/bin/env python

# Script to generate the data provided by Yuhui and used in testing demos

import numpy as np
import pandas as pd
from scipy.stats import  qmc


# TODO: Is there a better design to enforce objectivity?
# It makes it really clunky to pass it to the function and do the rotations there!
# TODO: if we don't need to memorize what this matrix should be, we could compute
# it on the fly inside the loss function and avoid all this. Let's talk about it
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
        # and the code below will always return for j=0. What am I missing?
        for j in range(3):
            subset = df1.iloc[j * sequence_length : (j + 1) * sequence_length]
            if len(subset) == sequence_length:
                # Standardize inputs to unit variance
                # TODO: is the mean of the input zero ? If not, doing this changes the mean !
                # TODO: dividing by stddev is done by Logarzo but I don't think this is a good normalization technique. Let's talk about it
                std_devs_in = subset[input_columns].std().to_numpy()
                std_devs_out = subset[output_columns].std().to_numpy()
                # Avoid division by zero
                # TODO: 1e-6 might be similar to the actual stddev so that could be a problem
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

def get_data(file_paths):
    # Read in and process file containing loading history for multiple tests
    # step, strain11, strain22, strain33, strain12, strain13, strain23, stress11, stress22, stress33, stress12, stress13, stress23, index, size, material
    # 1000 steps = [0 ; 999] vertically stacked for a given test, one step per row
    # 100 blocks (index = [0 ; 99]) of 1000 steps vertically stacked. What is the index representing ? One index per test ?
    # material = 0 for all tests
    # size = 30 for all tests
    # TODO: I have no idea what units stress, strain, size are!
    # TODO: Is there is any normalization?
    # TODO: Pandas uses its own `index`, so using a variable named `index` might be confusing. Consider changing name
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
    return X, y, R

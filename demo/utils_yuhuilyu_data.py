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


# TODO JBC: Looks like what was in this Jupyter block and the previous one could be
# manually chosen by the user to use either / or techniques to extract X and y
# from the data. To keep them and have them work in a regular script,
# I refactored them into functions
def extract_and_normalize_input_and_output(df_combined,
                                           sequence_length = 1000,
                                           input_n_features = 6,
                                           output_n_features = 6,
                                           input_columns = ["strain11", "strain22", "strain33", "strain12", "strain13", "strain23"],
                                           output_columns = ["stress11", "stress22", "stress33", "stress12", "stress13", "stress23"],
                                           normalization_style = 'interval',
                                           normalization_values = (0,1)):
    # Normalization can be: 'none', 'logarzo', 'interval', 'scale'
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
                inputs = subset[input_columns]
                outputs = subset[output_columns]

                match normalization_style:
                    case 'none': # No normalization
                        X[count] = inputs.to_numpy()
                        y[count] = outputs.to_numpy()
                    case 'logarzo': # Normalizes by per-feature standard deviation
                        # Logarzo et al. 2021  Eq. (16) - (17) https://doi.org/10.1016/J.CMA.2020.113482
                        # TODO: IMPORTANT DISCUSSION TO HAVE:
                        # I don't think the standardization *per feature* is a good idea:
                        # - 1. strain is tensor, is separating features relevant? should the dipersion measure employed be frame invariant? This can quickly get complex
                        # - 2. dividing by standard deviation changes the mean, and could change relative magnitude of strain component
                        #   - * if we generate eps_11 > eps_22 but eps_11 has more variance
                        #       we could end up with a "normalized" data where eps_22 > eps_11, which seems wrong
                        #       I think this might be mitigated by the fact we generate all
                        #       principal strains from the same GP covariance, but that might just be luck
                        std_devs_in = inputs.std().to_numpy()
                        std_devs_out = outputs.std().to_numpy()
                        # Avoid division by zero
                        # TODO: 1e-6 might be similar to the actual stddev so that could be a problem
                        std_devs_in[std_devs_in <= 0] = 1e-6
                        std_devs_out[std_devs_out <= 0] = 1e-6
                        X[count] = inputs.to_numpy() / std_devs_in
                        y[count] = outputs.to_numpy() / std_devs_out
                        # TODO: in prediction, how do you de-normalize?
                        # The normalization is relative to the input, and is
                        # computed over the entire sequence. You do not have
                        # that information avaialble in online prediction but
                        # the model is trained and expects normalized value
                    case 'interval': # Normalize over the chosen interval
                        if (len(normalization_values) != 2):
                            raise ValueError("must provide valid interval for normalization")
                        if (normalization_values[0] >= normalization_values[1]):
                            raise ValueError("normalization interval [l, u] must verify l < u")
                        l, u = normalization_values
                        # TODO: I think this should be avoided because the
                        # physical meaning of ZERO is lost. Same question as for
                        # Logarzo method: how do you de-normalize in online
                        # prediction? You don't know where the min/max will be!
                        X_min = inputs.min().to_numpy()
                        y_min = outputs.min().to_numpy()
                        X_max = inputs.max().to_numpy()
                        y_max = outputs.max().to_numpy()
                        X[count] = (u - l) * (inputs.to_numpy() - X_min) / (X_max - X_min) + l
                        y[count] = (u - l) * (outputs.to_numpy() - y_min) / (y_max - y_min) + l
                    case 'scale': # Normalize by scaling by a chosen factor
                        if (len(normalization_values) != 2):
                            raise ValueError("must provide scaling factor for both input and output")
                        if (normalization_values[0] <= 0 or normalization_values[1] <= 0):
                            raise ValueError("scaling factors must be positive")
                        scaling_in, scaling_out = normalization_values
                        # TODO: Same issues as [-1, 1]:
                        # physical meaning of ZERO is lost. Same question as for
                        # Logarzo method: how do you de-normalize in online
                        # prediction? You don't know where the min/max will be!
                        X_min = inputs.min().to_numpy()
                        y_min = outputs.min().to_numpy()
                        X_max = inputs.max().to_numpy()
                        y_max = outputs.max().to_numpy()
                        X[count] = inputs.to_numpy() * scaling_in
                        y[count] = outputs.to_numpy() * scaling_out
                count += 1
    return X, y


def get_data(file_paths, normalization_style = 'interval', normalization_values = (0,1)):
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

    X, y = extract_and_normalize_input_and_output(df_combined, normalization_style=normalization_style, normalization_values = normalization_values)
    return X, y, R

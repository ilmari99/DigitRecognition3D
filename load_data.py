# We have a 3D mnist dataset.
# The dataset contains csv files containing 3d coordinates of a finger position.
# The files are named as "stroke_1_<n>.csv", "stroke_2_<n>.csv" etc. The first number tells what the number is,
#and the second number is just an index for the stroke.

# Visualize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from sklearn.decomposition import PCA

DATA_ROOT = "./digits_3d_training_data/digits_3d/training_data"
def get_paths():
    paths = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(".csv"):
                paths.append(os.path.join(root, file))
    return paths

def read_data(pad_data = True):
    paths = get_paths()
    random.shuffle(paths)
    data = []
    labels = []
    for path in paths:
        data.append(pd.read_csv(path, header=None).to_numpy())
        labels.append(path.split("/")[-1].split("_")[1])
    labels = np.array(labels)
    # pad each sequence to the same length if pad_data is True
    if pad_data:
        pad_amount = max([d.shape[0] for d in data])
        for i in range(len(data)):
            data[i] = np.vstack((data[i],np.zeros((pad_amount-data[i].shape[0],3))))
        data = np.array(data)
    # If the data is not padded, it can not be converted to a numpy array since the sequences are of different length
    return data, labels

def visualize_data():
    """ Loop all paths, and plot the xyz coordinates in 3d plot"""
    paths = get_paths()
    random.shuffle(paths)
    for i,path in enumerate(paths):
        data = pd.read_csv(path, header=None)
        data = data.to_numpy()
        true_label = path.split("/")[-1].split("_")[1]
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2])
        plt.title(f"True label: {true_label}")
        plt.show()


def visualize_data_projected():
    """ Make each sequence to a 3d tensor, and project it to 2d using PCA. Then plot the 2d data.
    """
    paths = get_paths()
    random.shuffle(paths)
    for i, path in enumerate(paths):
        data = pd.read_csv(path, header=None)
        data = data.to_numpy()
        true_label = path.split("/")[-1].split("_")[1]
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pca = PCA(n_components=2)
        pca.fit(data)
        data = pca.transform(data)
        ax.scatter(data[:, 0], data[:, 1])
        plt.title(f"True label: {true_label}")
        plt.show()


        

if __name__ == "__main__":
    visualize_data_projected()

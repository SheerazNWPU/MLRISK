import math
import os
from os.path import join
from scipy.spatial.distance import mahalanobis
import numpy as np
from scipy.spatial.distance import hamming
from time import time
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm, trange
from sklearn.multioutput import MultiOutputClassifier
from numpy.linalg import LinAlgError
def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


# Calculate the center of high dimension data
def get_center(data_list):
    center = []

    for d in range(len(data_list[0])):  # d for dimension
        coordinate = 0

        for data in data_list:
            coordinate += data[d]

        coordinate /= len(data_list)
        center.append(coordinate)

    return center


def take_0(elem):
    return elem[0]


def point_min(list):
    min_index = list.index(min(list))

    for i in range(len(list)):

        if i == min_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


def calculate_multilabel_centroids(features, labels):
    # Assuming labels is a 2D binary matrix (n_samples, n_labels)
    n_labels = labels.shape[1]
    n_classes = 2 ** n_labels  # Total number of classes (2^n_labels)
    centroids = np.zeros((n_classes, features.shape[1]))

    for class_idx in range(n_classes):
        # Convert class index to binary representation
        class_binary = np.binary_repr(class_idx, width=n_labels)

        # Select samples where the labels match the current class
        class_mask = np.all(labels == np.array([int(bit) for bit in class_binary]), axis=1)
        class_features = features[class_mask]

        # Calculate the mean of these samples
        if class_features.size > 0:
            centroids[class_idx] = np.mean(class_features, axis=0)
        else:
            centroids[class_idx] = np.zeros(features.shape[1])  # Or handle empty classes differently

    return centroids

        
def get_inverse_covariance(cov_matrix, reg_param=1e-5):
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except LinAlgError:
        print("Covariance matrix is singular, using regularization.")
        cov_matrix += np.eye(cov_matrix.shape[0]) * reg_param  # Add small value to diagonal
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    return inv_cov_matrix

# Calculate the distance to each class center of every data
centers_binary = np.array([
          [0, 0, 0],  # Combination 0 (e.g., no labels)
          [0, 0, 1],  # Combination 1 (e.g., only third label)
          [0, 1, 0],  # Combination 2 (e.g., only second label)
          [0, 1, 1],  # Combination 3 (e.g., second and third labels)
          [1, 0, 0],  # Combination 4 (e.g., only first label)
          [1, 0, 1],  # Combination 5 (e.g., first and third labels)
          [1, 1, 0],  # Combination 6 (e.g., first and second labels)
          [1, 1, 1],  # Combination 7 (e.g., all labels)
      ])
def get_one_distance(
    layer, elem_name, num_class, targets_df, csv_dir, metrics, data_sets=["train", "val", "test"]
):
    start = time()

    # Calculate the class centers by train data
    coordinate_train = pd.read_csv(
        os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])),
        header=None,
    ).to_numpy()
    label = pd.read_csv(
        os.path.join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
    ).to_numpy()

    centers = calculate_multilabel_centroids(coordinate_train, label)

    for metric in metrics:
        print(f"--- Processing {metric} distance ---")

        inv_cov = None
        if metric == "mahalanobis":
            # Compute inverse covariance matrix for Mahalanobis distance
            cov_matrix = np.cov(coordinate_train, rowvar=False)
            inv_cov = get_inverse_covariance(cov_matrix, reg_param=1e-5)

        for data_set in data_sets:
            coordinates = pd.read_csv(
                os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_set)),
                header=None,
            ).to_numpy()
            labels = pd.read_csv(
                os.path.join(csv_dir, "targets_{}.csv".format(data_set)), header=None
            ).to_numpy()
            paths = pd.read_csv(
                os.path.join(csv_dir, "paths_{}.csv".format(data_set)), header=None
            ).to_numpy().flatten()
            predictions = pd.read_csv(
                os.path.join(csv_dir, "predictions_{}.csv".format(data_set)), header=None
            ).to_numpy()
            shorten_paths(paths)

            # Compute distance based on the metric
            distance_to_center = []
            if metric in ["euclidean", "cosine", "manhattan"]:
                distance_to_center = pairwise.pairwise_distances(
                    coordinates, centers, metric=metric
                ).tolist()

            elif metric == "mahalanobis":
                distance_to_center = [
                    [mahalanobis(coordinate, center, inv_cov) for center in centers]
                    for coordinate in coordinates
                ]

            elif metric == "hamming":
                distance_to_center = pairwise.pairwise_distances(
                    predictions, centers_binary, metric="hamming"
                ).tolist()
                #print(centers)

            # Changes made by Sheeraz 15/5/2024
            temp_labels = []
            temp_paths = []
            temp_distance_to_center = []
            for label, path, distance in zip(labels, paths, distance_to_center):
                if label[0] == 0:
                    cls = 0
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_distance_to_center.append(distance)
                if label[0] == 1:
                    cls = 1
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_distance_to_center.append(distance)
                if label[1] == 0:
                    cls = 2
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_distance_to_center.append(distance)
                if label[1] == 1:
                    cls = 3
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_distance_to_center.append(distance)
                if label[2] == 0:
                    cls = 4
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_distance_to_center.append(distance)
                if label[2] == 1:
                    cls = 5
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_distance_to_center.append(distance)

            temp_csv = []
            for i in range(len(temp_paths)):
                for j in range(num_class):
                    temp_csv.append(
                        [
                            "{}_{:0>3d}".format(temp_paths[i], j),
                            "{}".format(1 if j == temp_labels[i] else 0),
                            temp_distance_to_center[i][j],
                        ]
                    )

            # Use exec to assign temp_csv to a dynamically named variable
            exec("distance_to_center_{} = temp_csv".format(data_set))

        # Merge 3 csvs together
        distance_center_all = []
        for data_set in data_sets:
            exec("distance_center_all.extend(distance_to_center_{})".format(data_set))

        # Create the header of csv
        header = ["data", "label", "{}_{}_{}".format(elem_name, layer, metric)]

        # Save the final csv for this metric
        output_file = os.path.join(
            csv_dir, "{}_{}_one_{}.csv".format(elem_name, layer, metric)
        )
        distance_center_all.insert(0, header)
        pd.DataFrame(distance_center_all).to_csv(output_file, header=None, index=None)

    print("--- distance calculation {:.2f} s ---".format(time() - start))



if __name__ == "__main__":

    get_one_distance()

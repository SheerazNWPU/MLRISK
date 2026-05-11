import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
from numpy.linalg import inv
import os
from numpy.linalg import LinAlgError


def eval_distance(data_set, layer, labels, predictions, distances):
    # Evaluate prediction
    correct = 0
    for i in range(len(labels)):
        if np.all(predictions[i] == labels[i]):
            correct += 1

    acc = correct / len(labels)

    # Evaluate distance
    d_predictions = []

    for info in distances:
        # info = list(map(float, info))
        d_predictions.append(info.index(min(info)))

    d_correct = 0

    for i in range(len(labels)):
        #print(labels)
        #print(d_predictions)
        #print
        if np.all(d_predictions[i] == labels[i]):
            d_correct += 1

    d_acc = d_correct / len(labels)

    # Calculate different wrong
    evaluation = []

    for i in range(len(labels)):
        evaluation.append([labels[i], predictions[i], d_predictions[i]])

    p_wrong = d_wrong = 0

    for data_labels in evaluation:

        if np.all(data_labels[1] != data_labels[2]):

            if np.all(data_labels[1] == data_labels[0]):
                d_wrong += 1
            elif np.all(data_labels[2] == data_labels[0]):
                p_wrong += 1

    # print('Dataset: {}, Layer: {}'.format(data_set, layer))
    # print('Acc, D_Acc, d_right_p_wrong, p_right_d_wrong')
    print("{:.2f}, {:.2f}, {}, {}".format(acc * 100, d_acc * 100, p_wrong, d_wrong))
    # print('{:.2f}'.format(d_acc * 100))
# Other imports...
# Calculate the center of high dimension data
def get_center(data_list):
    center = []
    #print(data_list)
    for d in range(len(data_list[0])):  # d for dimension
        coordinate = 0

        for data in data_list:
            coordinate += data[d]

        coordinate /= len(data_list)
        center.append(coordinate)

    return center
def get_inverse_covariance(cov_matrix, reg_param=1e-5):
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except LinAlgError:
        print("Covariance matrix is singular, using regularization.")
        cov_matrix += np.eye(cov_matrix.shape[0]) * reg_param  # Add small value to diagonal
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    return inv_cov_matrix
def get_distance(
    layer, elem_name, num_class, csv_dir, metrics, data_sets=["train", "val", "test"]
):
    print("\n===== layer: {} =====".format(layer))
    print("Acc, D_Acc, d_right_p_wrong, d_wrong_p_right")

    # Calculate the class centers by train data
    coordinate_train = pd.read_csv(
        os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])),
        header=None,
    ).to_numpy()
    label = pd.read_csv(
        os.path.join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
    ).to_numpy()
    n_labels = len(label[0])
    train_cls = []
    for cls in range(num_class):
        coordinate_cls = []
        
        # Convert class index to binary representation
        class_binary = np.binary_repr(cls, width=n_labels)
        class_mask = np.array([int(bit) for bit in class_binary])
    
        for i in range(len(label)):
            if np.all(label[i] == class_mask):
                coordinate_cls.append(coordinate_train[i])
        
        train_cls.append(coordinate_cls)

    centers = []
    for i in range(len(train_cls)):
        centers.append(get_center(train_cls[i]))

    # Compute inverse covariance matrix for Mahalanobis distance
    cov_matrix = np.cov(coordinate_train.T)
    inv_cov_matrix = get_inverse_covariance(cov_matrix, reg_param=1e-5)
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
    for metric in metrics:
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
    
            # Calculate distance to centers using selected metric
            if metric == "mahalanobis":
                distance_to_center = [
                    [mahalanobis(coord, center, inv_cov_matrix) for center in centers]
                    for coord in coordinates
                ]
            elif metric == "hamming":
                # Ensure the coordinates and centers are binary
                #coordinates_binary = np.round(coordinates).astype(int)
                #centers_binary = np.round(centers).astype(int)
                #print(coordinates_binary)
                #print(centers_binary)
                distance_to_center = pairwise.pairwise_distances(
                    predictions, centers_binary, metric="hamming"
                ).tolist()
            else:
                distance_to_center = pairwise.pairwise_distances(
                    coordinates, centers, metric=metric
                ).tolist()
    
            # Evaluate distance and save results
            eval_distance(data_set, layer, labels, predictions, distance_to_center)
    
            for i in range(len(distance_to_center)):
                distance_to_center[i].insert(0, paths[i])
                distance_to_center[i].insert(1, labels[i])
    
            exec("distance_to_center_{} = distance_to_center".format(data_set))
    
        # Save results for each metric
        distance_center_all = []
        for data_set in data_sets:
            exec("distance_center_all.extend(distance_to_center_{})".format(data_set))
    
        header = ["data", "label"]
        for i in range(num_class):
            header.append("{}_{}_class_{:0>3d}_{}".format(elem_name, layer, i, metric))
    
        distance_center_all.insert(0, header)
        pd.DataFrame(distance_center_all).to_csv(
            os.path.join(csv_dir, "{}_{}_{}.csv".format(elem_name, layer, metric)),
            header=None,
            index=None,
        )

    print(f"Distance calculation with {metric} completed and saved.")



# Run the function for different metrics
if __name__ == "__main__":

    for layer in layers:
        for cnn in cnns:
            print("===== CNN: {} =====".format(cnn))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(cnn)
            num_class = (
                int(
                    pd.read_csv(join(csv_dir, "predictions_train.csv"), header=None)
                    .to_numpy()
                    .flatten()[-1]
                )
                + 1
            )

            # Run with multiple metrics
            for metric in metrics + ["mahalanobis", "hamming"]:
                get_distance(layer, elem_name, num_class, csv_dir, metric, data_sets)

    print("All distance calculations done.")

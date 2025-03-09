
import multiprocessing
import os
from functools import partial
from itertools import combinations
from os.path import join

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from tqdm import tqdm, trange
from time import time


def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


def point_max(list):
    max_index = list.index(max(list))

    for i in range(len(list)):

        if i == max_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


def get_knn(k, list, label_train):
    list_sorted = sorted(list)
    knn = []

    for i in range(k):
        index = list.index(list_sorted[i])
        knn.append(label_train[index])

    return knn


def eval_knn(data_set, layer, labels, predictions, count):
    # Evaluate prediction
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1

    acc = correct / len(labels)

    # Evaluate distance
    k_predictions = []

    for info in count:
        # info = list(map(int, info))
        k_predictions.append(info.index(max(info)))

    k_correct = 0

    for i in range(len(labels)):
        if k_predictions[i] == labels[i]:
            k_correct += 1

    k_acc = k_correct / len(labels)

    # Calculate different wrong
    evaluation = []

    for i in range(len(labels)):
        evaluation.append([labels[i], predictions[i], k_predictions[i]])

    p_wrong = k_wrong = 0

    for data_labels in evaluation:

        if data_labels[1] != data_labels[2]:

            if data_labels[1] == data_labels[0]:
                k_wrong += 1
            elif data_labels[2] == data_labels[0]:
                p_wrong += 1

    
  
    # print('{:.2f}'.format(k_acc * 100))


# Calculate the knn_count to each class center of every data
def get_one_knn_count(
    k_list,
    layer,
    elem_name,
    num_class,
    targets_df,    
    csv_dir,
    metric,
    data_sets=["train", "val", "test"],
):
    for k in k_list:
        start = time()

        # Get knn_count for all data_sets
        count_all = []
        for data_set in data_sets:
            labels_train = (
                pd.read_csv(
                    join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
                )
                .to_numpy()#.flatten()   ##Can be a problem
            )
            distribution_train = pd.read_csv(
                join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])),
                header=None,
            ).to_numpy()

            paths = (
                pd.read_csv(join(csv_dir, "paths_{}.csv".format(data_set)), header=None)
                .to_numpy()
                .flatten()
            )
            shorten_paths(paths)
            labels = (
                pd.read_csv(
                    join(csv_dir, "targets_{}.csv".format(data_set)), header=None
                )
                .to_numpy()
            )
            distribution = pd.read_csv(
                join(csv_dir, "distribution_{}_{}.csv".format(layer, data_set)),
                header=None,
            ).to_numpy()

            # Get pairwise_distances for all
            p_d = pairwise.pairwise_distances(distribution, distribution_train, metric=metric)

            
            # Sheeraz Made Changes 8/5/2024
            # Get knn count for all
            s_p_d = np.sort(p_d)
            knn = np.zeros([len(p_d), k, 6], dtype=int)  # 6 represents the total number of classes
            
            for i in range(len(p_d)):
                for j in range(k):
                    if data_set == 'train':
                        idx = np.where(p_d[i] == s_p_d[i, j+1])[0][0]
                        labels_t = labels_train[idx]
                        for l in range(3):  # Iterate over the three labels
                            class_idx_0 = l * 2  # Index for label=0
                            class_idx_1 = l * 2 + 1  # Index for label=1
                            knn[i, j, class_idx_0] = int(labels_t[l] == 0)  # Set 1 if label=0, else 0
                            knn[i, j, class_idx_1] = int(labels_t[l] == 1)  # Set 1 if label=1, else 0
                    else:
                        idx = np.where(p_d[i] == s_p_d[i, j])[0][0]
                        labels_t = labels_train[idx]
                        for l in range(3):  # Iterate over the three labels
                            class_idx_0 = l * 2  # Index for label=0
                            class_idx_1 = l * 2 + 1  # Index for label=1
                            knn[i, j, class_idx_0] = int(labels_t[l] == 0)  # Set 1 if label=0, else 0
                            knn[i, j, class_idx_1] = int(labels_t[l] == 1)  # Set 1 if label=1, else 0
            
            count = [np.sum(knn[i], axis=0).tolist() for i in range(len(knn))]  # Sum over k neighbors for each class
            
           
            
            # Changes made by Sheeraz 15/5/2024
            temp_labels = []
            temp_paths = []
            temp_count = []
            for label, path, coun in zip(labels, paths, count):
                if label[0] == 0:
                    cls = 0
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_count.append(coun)
                if label[0] == 1:
                    cls = 1
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_count.append(coun)
                if label[1] == 0:
                    cls = 2
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_count.append(coun)
                if label[1] == 1:
                    cls = 3
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_count.append(coun)
                if label[2] == 0:
                    cls = 4
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_count.append(coun)
                if label[2] == 1:
                    cls = 5
                    temp_labels.append(cls)
                    temp_paths.append(path)
                    temp_count.append(coun)

            temp_csv = []
            
            # Changes made by Sheeraz 7/5/2024
            
            #print(temp_count)
            for i in range(len(temp_paths)):
            
                for j in range(num_class):    
                    temp_csv.append(
                       [
                            "{}_{:0>3d}".format(temp_paths[i], j),
                            "{}".format(1 if j == temp_labels[i] else 0),
                            temp_count[i][j],
                        ]
                    )
          

            count_all.extend(temp_csv)

        # Create the header of csv
        header = ["data", "label", "{}_{}_count{}".format(elem_name, layer, k)]

        # Save the final csv
        count_all.insert(0, header)
        pd.DataFrame(count_all).to_csv(
            os.path.join(csv_dir, "{}_{}_one_knn{}.csv".format(elem_name, layer, k)),
            header=None, index=None
        )

        print("--- knn_{} {:.2f} s ---".format(k, time() - start))


if __name__ == "__main__":

    get_one_knn_count()

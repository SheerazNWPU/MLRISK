
import os
from os.path import join

import numpy as np
from time import time
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from sklearn.neighbors import NearestCentroid
import pickle
from scipy.special import softmax
from tqdm import tqdm, trange


def shorten_paths(paths):
    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


# Calculate the distance to each class center of every data
def get_probability(
    layer, elem_name, num_class, csv_dir, gml_dir,  data_sets=["train", "val", "test"]
):
    start = time()

    for data_set in data_sets:
        # Read coordinate and label of data
        # coordinates = pd.read_csv(
        #     os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_set)),
        #     header=None,
        # ).to_numpy()
        # labels = (
        #     pd.read_csv(
        #         os.path.join(csv_dir, "targets_{}.csv".format(data_set)), header=None
        #     )
        #     .to_numpy()
        #     .flatten()
        # )
        # paths = (
        #     pd.read_csv(
        #         os.path.join(csv_dir, "paths_{}.csv".format(data_set)), header=None
        #     )
        #     .to_numpy()
        #     .flatten()
        # )

        coordinates = np.array(
            pickle.load(open(join(csv_dir, 'distribution_{}_{}.pkl'.format(layer, data_set)), 'rb+')))
        labels = np.array(
            pickle.load(open(join(csv_dir, 'targets_{}.pkl'.format(data_set)), 'rb+'))).flatten()
        paths = np.array(
            pickle.load(open(join(csv_dir, 'paths_{}.pkl'.format(data_set)), 'rb+'))).flatten()
        shorten_paths(paths)

        # Calculate the distance to each class center of every data
        probability = softmax(coordinates, axis=1).tolist()

        # Insert path and label of data to csv
        for i in range(len(probability)):
            probability[i].insert(0, paths[i])
            probability[i].insert(1, labels[i])

        exec("probability_{} = probability".format(data_set))

    # Merge 3 csvs together
    probability_all = []

    for data_set in data_sets:
        exec("probability_all.extend(probability_{})".format(data_set))

    # Create the header of csv
    header = ["data", "label"]

    for i in range(num_class):
        header.append("{}_{}_class_{:0>3d}_probability".format(elem_name, layer, i))

    # Save the final csv
    probability_all.insert(0, header)
    with open(join(gml_dir, "{}_{}_prob.pkl".format(elem_name, layer)), 'wb+') as pkl:
        pickle.dump(probability_all, pkl)

    print("--- prob {:.2f} s ---".format(time() - start))


if __name__ == "__main__":
        get_probability()

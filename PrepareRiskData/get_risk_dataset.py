import os
import shutil
from os.path import join

import numpy as np
import pandas as pd
from get_one_risk_info import get_one_risk_info
from get_risk_info import get_risk_info


def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



data_dirs = ['Multilabel']  # dataset dir
data_sets = ['train', 'val', "test"]
cnns = ['50'] #'r101', 'r50','d169','CCT'    # multiple CNNs used to get features
layers =['xc','x4', 'bay', 'moe'] # ['x4', 'xc', 'mi', 'f', 'maxif', 'ka', 'pet', 'kt', 'sp']
elem_name_str = "{}"
elems = ["cosine", 'mahalanobis', 'hamming'] # ["distance", 'knn1', 'knn8','xs5','xs3']
csv_dir_str = "/YourSavePath/result_archive/{}{}"  # where you save the distribution csv
# csv_dir_str = "/home/ssd0/lfy/result_archive/tvt/{}_{}"  # where you save the distribution csv
k_list = [5,7] # [1, 8]
archive_dir = "/YourSavePath/result_archive/risk_elem/"  # path to save risk feature csv
note = ''  # additional note to add in save folder, e.g., save 'fgvc_100_test' at archive_dir

print(csv_dir_str)

# get risk elem
if __name__ == "__main__":

    for data_dir in data_dirs:
        print('\n===== {} ====='.format(data_dir))
        my_mkdir(join(archive_dir, data_dir))
        my_mkdir(join(archive_dir, data_dir, "risk_dataset{}".format(note)))
        my_mkdir(join(archive_dir, data_dir, "softmax"))
        my_mkdir(join(archive_dir, data_dir, "DBLP-Scholar"))
        my_mkdir(join(archive_dir, data_dir, "DBLP-Scholar", '325'))

        get_risk_info(
            data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, note
        )
        get_one_risk_info(
            data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list
        )

        # copy softmax to final save dir
        for cnn in cnns:
            my_mkdir(join(archive_dir, data_dir, "softmax", "{}".format(cnn)))
            for data_set in data_sets:
                shutil.copy(
                    join(
                        csv_dir_str.format(data_dir, cnn),
                        "distribution_xc_{}.csv".format(data_set),
                    ),
                    join(
                        archive_dir,
                        data_dir,
                        "softmax",
                        "{}".format(cnn),
                        "distribution_xc_{}.csv".format(data_set),
                    ),
                )

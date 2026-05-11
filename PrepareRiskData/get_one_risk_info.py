from os.path import join

import numpy as np
import pandas as pd
from get_one_distance import get_one_distance
from get_one_knn_count import get_one_knn_count


def get_one_risk_info(
    data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list
):
    # get risk elem
    for layer in layers:
        for cnn in cnns:
            print("=== geting one_risk_info of {}_{} ===".format(cnn, layer))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(data_dir, cnn)
            # Read the targets CSV file
            targets_df = pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])))
            
            # Get the number of classes for each label
            num_classes = [targets_df[label].nunique() for label in targets_df.columns]

            # Display the number of classes for each label
            print("Number of classes for each label:", num_classes)
            
            num_class =sum(num_classes)
           
            
            get_one_distance(layer, elem_name, num_class, targets_df, csv_dir, elems, data_sets)
            

    # Merge csv
    csv_path_list = []
    for cnn in cnns:
        for layer in layers:
            for elem in elems:
                
                if elem == 'fangcha':
                    if layer == 'x4':
                        continue
                    if cnn == 'CCT':
                        continue
                if elem == 'xs8' or elem == 'xs1' or elem == 'xs3' or elem == 'xs5':
                    if layer == 'x4':
                        continue
                    if cnn == 'CCT':
                        continue
                if elem == 'paddingdis':
                    if cnn == 'CCT':
                        continue
                if elem == 'padknn8' or elem == 'padknn1':
                    if cnn == 'CCT':
                        continue
                if elem=='xsdis':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                if elem == 'all3' or elem == 'all5':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                csv_path_list.append(
                    join(
                        csv_dir_str.format(data_dir, cnn),
                        "{}_{}_one_{}.csv".format(cnn, layer, elem),
                    )
                )
    
    #print(csv_path_list[0])
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]
    #print(csv_path_list)
    for csv_path in csv_path_list:
        print(csv_path)
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, -1:]
        #print(len(all_info))
        #print(len(csv))
        all_info = np.hstack((all_info, csv))

    pd.DataFrame(all_info).to_csv(
        "/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{}/DBLP-Scholar/pair_info_more.csv".format(data_dir), #change path to the save distribution
        header=None,
        index=None,
    )

    all_info = all_info.tolist()
    # Load the image IDs for each set
    train_ids = np.loadtxt('/YourDatasetPath/BCNB/dataset-splitting/train_id.txt', dtype=str)
    val_ids = np.loadtxt('/YourDatasetPath/BCNB/dataset-splitting/val_id.txt', dtype=str)
    test_ids = np.loadtxt('/YourDatasetPath/BCNB/dataset-splitting/test_id.txt', dtype=str)
    
    # Updated Sheeraz 2/5/2024
    # Create a dictionary to map image IDs to their corresponding set
    id_to_set = {}
    for image_id in train_ids:
        id_to_set[image_id] = 'train'
    for image_id in val_ids:
        id_to_set[image_id] = 'val'
    for image_id in test_ids:
        id_to_set[image_id] = 'test'
    
    for data_set in ['train']:
        temp_csv = [all_info[0]]
        for line in all_info[1:]:
            image_id = line[0].split("/")[-1].split(".")[0]  # Extract the image ID from the file path
            if id_to_set.get(image_id, None) == data_set:
                temp_csv.append(line)
    
        pd.DataFrame(temp_csv).to_csv(
            "/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{}/DBLP-Scholar/325/{}.csv".format( #change path to the save distribution
                data_dir, data_set
            ),
            header=None,
            index=None,
        )


if __name__ == "__main__":

    get_one_risk_info()

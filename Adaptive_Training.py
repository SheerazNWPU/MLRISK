from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
from glob import glob
from os.path import join
import pandas as pd
import numpy as np
import time
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from  Folder import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix
import torch, random
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, hamming_loss
import torch.distributed as dist
from Densenet import densenet121
from Densenet import densenet169
from Densenet import densenet201
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from PIL import Image
#import albumentations as A
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
# import calibration as cb
from utils import *
from torch.utils.data import Dataset
from risk_one_rule import risk_dataset
from risk_one_rule import risk_torch_model
import risk_one_rule.risk_torch_model as risk_model
from common import config as config_risk
from torch.nn.functional import softmax, sigmoid
from scipy.special import softmax

import csv

cfg = config_risk.Configuration(config_risk.global_data_selection, config_risk.global_deep_learning_selection)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

"""Seed and GPU setting"""
seed = (int)(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.cuda.manual_seed(seed)

cudnn.benchmark = True
cudnn.deterministic = True

## Allow Large Images
Image.MAX_IMAGE_PIXELS = None

def hamming_loss(y_true, y_pred):
    # Ensure y_true and y_pred are tensors
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Convert predictions to binary if they are probabilities
    y_pred = (y_pred > 0.5).float()

    # Calculate the Hamming loss
    loss = torch.mean((y_true != y_pred).float())
    return loss

class DropConnectWrapper(nn.Module):
    def __init__(self, module, drop_prob):
        super(DropConnectWrapper, self).__init__()
        self.module = module
        self.drop_prob = drop_prob

    def forward(self, x):
        # Drop connections with probability drop_prob
        if self.training:
            mask = torch.bernoulli(torch.ones_like(self.module.weight) * (1 - self.drop_prob))
            weight = self.module.weight * mask
        else:
            weight = self.module.weight

        out = F.linear(x, weight, self.module.bias)
        return out
# Define the dataset class
class LabelDependencyGCN(nn.Module):
    def __init__(self, num_labels, hidden_dim=64):
        super(LabelDependencyGCN, self).__init__()
        self.conv1 = GCNConv(num_labels, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, num_labels)  # Second GCN layer (back to label space)
    
    def forward(self, logits, edge_index):
        """
        Refines CNN logits using label dependencies.
        
        Args:
            logits (torch.Tensor): CNN output logits of shape [batch_size, num_labels].
            edge_index (torch.Tensor): Edge list tensor defining label dependencies.

        Returns:
            torch.Tensor: Refined logits of shape [batch_size, num_labels].
        """
        x = F.relu(self.conv1(logits, edge_index))  # Apply GCN layer with ReLU activation
        x = self.conv2(x, edge_index)  # Apply second GCN layer
        return x
class LabelTransformer(nn.Module):
    def __init__(self, num_labels, hidden_dim=64, num_heads=4, num_layers=2):
        super(LabelTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.embedding = nn.Linear(num_labels, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        refined_output = self.output_layer(x)
        return refined_output

class AdaptiveLabelLearningRateScheduler:
    def __init__(self, optimizer, initial_lr, label_count, patience=2, factor=0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.label_count = label_count
        self.patience = patience
        self.factor = factor
        self.performance_history = {label: [] for label in range(label_count)}

    def step(self, label_performance):
        for label in range(self.label_count):
            self.performance_history[label].append(label_performance[label])
            if len(self.performance_history[label]) > self.patience:
                self.performance_history[label].pop(0)
            # Example condition: if performance has not improved, reduce the learning rate
            if len(self.performance_history[label]) == self.patience:
                if self.performance_history[label][-1] < min(self.performance_history[label]):
                    new_lr = self.optimizer.param_groups[label]['lr'] * self.factor
                    #adaptive LR
                    print(new_lr)
                    self.optimizer.param_groups[label]['lr'] = max(new_lr, 1e-6)

    def reset(self):
        for label in range(self.label_count):
            self.optimizer.param_groups[label]['lr'] = self.initial_lr[label]

class CustomDataset(Dataset):
    def __init__(self, image_dir, metadata_file, ids_file, transform=None):
        self.metadata = pd.read_excel(metadata_file)
        self.image_ids = np.loadtxt(ids_file, dtype=str)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx] + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
    
        label_er = self.metadata['ER'].map({'Positive': 1, 'Negative': 0}).values[idx]
        label_pr = self.metadata['PR'].map({'Positive': 1, 'Negative': 0}).values[idx]
        label_her2 = self.metadata['HER2'].map({'Positive': 1, 'Negative': 0}).values[idx]
    
        label = [label_er, label_pr, label_her2]
    
        return image, torch.FloatTensor(label), image_path
        
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        
        # Create a tensor for true distribution
        true_dist = torch.full_like(pred, self.smoothing / (self.cls - 1))
        
        # Update only the indices of the target
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Define the RegularizedLoss class
class RegularizedLoss(nn.Module):
    def __init__(self, base_loss_fn, co_occurrence_matrix, lambda_reg=0.1):
        super(RegularizedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.co_occurrence_matrix = co_occurrence_matrix
        self.lambda_reg = lambda_reg

    def forward(self, outputs, targets):
        base_loss = self.base_loss_fn(outputs, targets)
        regularization_term = self.lambda_reg * self.compute_regularization(outputs, targets)
        return base_loss + regularization_term

    def compute_regularization(self, outputs, targets):
        # Ensure outputs and targets are 2D tensors with shape (batch_size, num_classes)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
    
        batch_size = outputs.size(0)
        reg_loss = 0.0
    
        # Iterate over all pairs of labels
        for i in range(self.co_occurrence_matrix.size(0)):
            for j in range(self.co_occurrence_matrix.size(1)):
                reg_loss += self.co_occurrence_matrix[i, j] * torch.sum(targets[:, i] * outputs[:, j])
    
        # Normalize by batch size
        return reg_loss / batch_size


def compute_and_normalize_co_occurrence_matrix(train_labels):
    """
    Computes and normalizes the co-occurrence matrix from the training labels.

    Args:
        train_labels (torch.Tensor): Tensor of shape (num_samples, num_classes) with binary labels.

    Returns:
        torch.Tensor: Normalized co-occurrence matrix.
    """
    #train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    # Calculate co-occurrence matrix by matrix multiplication
    co_occurrence_matrix = torch.mm(train_labels.T, train_labels)
    
    # Normalize by the maximum value in the matrix to get values between 0 and 1
    normalized_co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.max()
    
    return normalized_co_occurrence_matrix

def output_risk_scores(file_path, id_2_scores, label_index, ground_truth_y, predict_y):
    op_file = open(file_path, 'w+', 1, encoding='utf-8')
    #print(op_file)
    for i in range(len(id_2_scores)):
        #print("CHECK")
        _id = id_2_scores[i][0]
        _risk = id_2_scores[i][1]
        _label_index = label_index.get(_id)
        _str = "{}, {}, {}, {}".format(ground_truth_y[_label_index],
                                       predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()
    return True
def collect_true_labels(dataloader):
    all_labels = []
    for batch_idx, (inputs, targets, paths) in enumerate(dataloader):
        all_labels.append(targets.numpy())  # Collect targets from each batch
    return np.concatenate(all_labels, axis=0)  # Concatenate all labels into one array

def prepare_data_4_risk_data():
    """
    first, generate , include all_info.csv, train.csv, val.csv, test.csv.
    second, use csvs to generate rules. one rule just judge one class
    :return:
    """
    #print('hello')
    #print(cfg)
    train_data, validation_data, test_data = risk_dataset.load_data(cfg)
    return train_data, validation_data, test_data

def prepare_data_4_risk_model(train_data, validation_data, test_data):

    rm = risk_torch_model.RiskTorchModel()
    rm.train_data = train_data
    rm.validation_data = validation_data
    rm.test_data = test_data
    return rm

# --------------------------------------------------------------------------------

class Adaptive_Trainer():

    # ---- Train the densenet network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- class_num - number of output classes
    # ---- batch_size - batch size
    # ---- nb_epoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the model_path file
    # ---- model_path - if not None loads the model and continues training

    def train(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, class_num, batch_size, nb_epoch,
              transResize, transCrop, launchTimestamp, val_num, store_name, model_path,  start_epoch=0,resume=False):
        save_name = os.path.join('/home/Gul/SG/sheeraz/risk_val_pmg_result/', str(val_num), store_name.split('/')[-1],
                                 str(seed))
        print(save_name)
        if (not os.path.exists(save_name)):
            os.makedirs(save_name)

        # setup output
        exp_dir = save_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            os.stat(exp_dir)
        except:
            os.makedirs(exp_dir)

        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        print(nnArchitecture)

  
        model_zoo = {'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152, 'wrn50':wide_resnet50_2, 'wrn101':wide_resnet101_2,
                      'd121':densenet121, 'd169':densenet169, 'd201':densenet201, 'eb4': efficientnet_b4,
                         'rx50': resnext50_32x4d}
        model = model_zoo['r50'](pretrained=True).cuda()
        in_features = model.fc.in_features

        model.fc = nn.Linear(in_features, class_num)
  


        lr_begin = 0.0005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)

        
        if model_path != None:
            try:
                try: model.load_state_dict(torch.load(model_path)['model'])
                except: model.load_state_dict(torch.load(model_path))
            except:
                model = torch.nn.DataParallel(model, device_ids=[0, 1])
                try: model.load_state_dict(torch.load(model_path)['model'])
                except: model.load_state_dict(torch.load(model_path))
        
        #model = nn.DataParallel(model, device_ids=[0, 1])
        model.cuda()
        adaptive_lr_scheduler = AdaptiveLabelLearningRateScheduler(optimizer=optimizer, initial_lr=[lr_begin]*class_num, label_count=class_num)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,16,2)

        # ---- TRAIN THE NETWORK
        train_data, val_data, test_data = prepare_data_4_risk_data()
        #print(train_data.true_labels)
        #print(train_data)
        risk_data = [train_data, val_data, test_data]

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   

        transformList = []
        transformList.append(transforms.Resize(256))
   
        transformList.append(transforms.FiveCrop(224))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test = transforms.Compose(transformList)
  
        testset = CustomDataset(image_dir='/home/Gul/Datasets/BCNB/BCNB/WSIs/',
                              metadata_file='/home/Gul/Datasets/BCNB/BCNB/patient-clinical-data.xlsx',
                              ids_file='/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/test_id.txt',
                              transform=transform_test)
        valset = CustomDataset(image_dir='/home/Gul/Datasets/BCNB/BCNB/WSIs/',
                              metadata_file='/home/Gul/Datasets/BCNB/BCNB/patient-clinical-data.xlsx',
                              ids_file='/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/val_id.txt',
                              transform=transform_test)
                                                    
        
        TestDataLoader = torch.utils.data.DataLoader(
            testset,
            batch_size=16,
            shuffle=True,
            num_workers=2,  # Increase this based on your CPU cores
            pin_memory=True  # Faster data transfer to GPU
        )
        LSLLoss = LabelSmoothingLoss(class_num, 0.1)
        LSLLoss1 = LabelSmoothingLoss(class_num, 0.1)
        LSLLoss2 = LabelSmoothingLoss(class_num, 0.1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        max_test_acc=0

        for epochID in range(0, nb_epoch):
            #Use different models for different epochs
            print('/home/15t/Gul/Datasets/{}/train'.format('BCNB'))
            _,train_pre=Adaptive_Trainer.train_t('/home/15t/Gul/Datasets/{}/WSIs/'.format(store_name),model, 'RESNET-101', class_num, False, 1,
                                                        256,244)
            _,val_pre=Adaptive_Trainer.Valtest('/home/15t/Gul/Datasets/{}//WSIs/'.format(store_name),model, 'RESNET-101',class_num, False, 1,
                                                        256,244)
            _,test_pre, test_pre_f1 =Adaptive_Trainer.test('/home/15t/Gul/Datasets/{}/WSIs/'.format(store_name),model, 'RESNET-101',class_num, False, 1,
                                                        256,244)

            my_risk_model = prepare_data_4_risk_model(risk_data[0], risk_data[1], risk_data[2])
            
            
            # Initialize empty tensors
            train_one_pre = torch.empty((0, class_num), dtype=torch.float64)
            val_one_pre = torch.empty((0, class_num), dtype=torch.float64)
            test_one_pre = torch.empty((0, class_num), dtype=torch.float64)
            
            # Function to process predictions
            def process_predictions(predictions):
                # Extract the maximum values for each pair of predictions
                max_values = torch.stack([torch.max(predictions[:, i:i+2], dim=1).values for i in range(0, 6, 2)], dim=1)
                return max_values
            
            # Function to process labels
            def process_labels(predictions):
                # Compare pairs and assign 1 or 0 based on which value is greater
                labels = torch.stack([(predictions[:, i] > predictions[:, i+1]).long() for i in range(0, 6, 2)], dim=1)
                return labels
            
            # Process train, validation, and test predictions
            train_max_values = process_predictions(train_pre)
            val_max_values = process_predictions(val_pre)
            test_max_values = process_predictions(test_pre)
            
            # Concatenate the processed predictions
            train_one_pre = torch.cat((train_one_pre, train_max_values), dim=0).cpu().numpy()
            val_one_pre = torch.cat((val_one_pre, val_max_values), dim=0).cpu().numpy()
            test_one_pre = torch.cat((test_one_pre, test_max_values), dim=0).cpu().numpy()
            
            # Process train, validation, and test labels
            train_labels = process_labels(train_pre)
            val_labels = process_labels(val_pre)
            test_labels = process_labels(test_pre)
            
            # Move labels to the appropriate device (assuming device is defined)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_labels = train_labels.to(device)
            val_labels = val_labels.to(device)
            test_labels = test_labels.to(device)
            
            my_risk_model.train(train_one_pre, val_one_pre, test_one_pre, train_pre.cpu().numpy(),
                                     val_pre.cpu().numpy(),
                                     test_pre.cpu().numpy(), train_labels, val_labels, test_labels, epochID)
            my_risk_model.predict(test_one_pre, test_pre.cpu().numpy(), )

            test_num = my_risk_model.test_data.data_len
            test_ids = my_risk_model.test_data.data_ids
            test_pred_y = test_labels
            test_true_y = my_risk_model.test_data.true_labels
            risk_scores = my_risk_model.test_data.risk_values
 
            
            

            id_2_label_index = dict()
            id_2_VaR_risk = []
            for i in range(test_num):
                id_2_VaR_risk.append([test_ids[i], risk_scores[i]])
                id_2_label_index[test_ids[i]] = i
            ## Main Point for risk sorting
            id_2_VaR_risk = sorted(id_2_VaR_risk, key=lambda item: sum(item[1]), reverse=True)
            
            print('this is epoch: {}'.format(epochID))
            output_risk_scores('{}/risk_score_epoch_{}.txt'.format(exp_dir, epochID), id_2_VaR_risk, id_2_label_index, test_true_y, test_pred_y)
            
            all_id_2_risk_desc = []
            num_labels = len(risk_scores[0])
            for label_idx in range(num_labels):
                id_2_risk = []
                for i in range(test_num):
                    test_pred = test_one_pre[i][label_idx]  # Prediction for the current label
                    m_label = test_pred_y[i][label_idx]     # Predicted label
                    t_label = test_true_y[i][label_idx]     # True label
                    if m_label == t_label:
                        label_value = 0.0
                    else:
                        label_value = 1.0
                    id_2_risk.append([test_ids[i], 1 - test_pred])
                id_2_risk_desc = sorted(id_2_risk, key=lambda item: item[1], reverse=True)
                all_id_2_risk_desc.extend(id_2_risk_desc)
            
            output_risk_scores('{}/base_score_epoch_{}.txt'.format(exp_dir, epochID), all_id_2_risk_desc, id_2_label_index, test_true_y, test_pred_y)
                                

            budgets = [10, 20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            risk_correct = [0] * len(budgets)
            base_correct = [0] * len(budgets)
            for i in range(test_num):
                for budget in range(len(budgets)):
                    if i < budgets[budget]:
                        pair_id = id_2_VaR_risk[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            risk_correct[budget] += 1
                        pair_id = id_2_risk_desc[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            base_correct[budget] += 1


            risk_loss_criterion = risk_model.RiskLoss(my_risk_model)
            risk_loss_criterion = risk_loss_criterion.cuda()

            rule_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_discrete(), dtype=torch.float64).cuda()
            machine_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_continue(), dtype=torch.float64).cuda()
            rule_activate = torch.tensor(my_risk_model.test_data.get_rule_activation_matrix(),
                                         dtype=torch.float64).cuda()
            machine_activate = torch.tensor(my_risk_model.test_data.get_prob_activation_matrix(),
                                            dtype=torch.float64).cuda()
            machine_one = torch.tensor(my_risk_model.test_data.machine_label_2_one, dtype=torch.float64).cuda()
            risk_y = torch.tensor(my_risk_model.test_data.risk_labels, dtype=torch.float64).cuda()


            test_ids = my_risk_model.test_data.data_ids
            test_ids_dict = dict()
            for ids_i in range(len(test_ids)):
                test_ids[ids_i] = os.path.basename(
                    test_ids[ids_i])
                test_ids_dict[test_ids[ids_i]] = ids_i

            del my_risk_model

            data_len = len(risk_y)
    
            model.train()
            best_performance = None
            best_model_state = None
            risk_labels = None
            epoch_loss = 0
            all_confidences=[]
            sum_confidence_high = 0.0
            sum_confidence_medium = 0.0
            num_batches_high = 0
            num_batches_medium = 0
            #out_uncertain=None
            for batch_idx, (inputs, targets, paths) in enumerate(TestDataLoader):

                optimizer.zero_grad()

                idx = batch_idx
                if inputs.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)

  
                index = []

                # we just need class_name and image_name
                paths = list(paths)
                for path_i in range(len(paths)):
                    paths[path_i] = os.path.basename(
                        paths[path_i])
                    # print(paths[path_i])
                    #index.append(test_ids_dict.get(paths[path_i], -1))
                    index.append(test_ids_dict[paths[path_i]])
                #               print(index)

                test_pre_batch = test_pre[index]
                rule_mus_batch = rule_mus[index]
                machine_mus_batch = machine_mus[index]
                rule_activate_batch = rule_activate[index]
                machine_activate_batch = machine_activate[index]
                machine_one_batch = machine_one[index]

                # optimizer.zero_grad()
                # _, _, _, output_concat, _, _ = net(inputs)
                chex=1
                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)

                except:
                    xc = model(inputs)

                if chex == 1:
                    xc = xc.cuda().squeeze().view(bs, n_crops, -1).mean(1)
                
                batch_confidence_high = xc.mean().item()
    
                # Update running sum and count
                sum_confidence_high += batch_confidence_high
                num_batches_high += 1
                

                correlation_matrix = torch.corrcoef(xc.T)  # Correlation between labels

                # Step 2: Define a threshold for strong dependencies
                threshold = 0.5
                strong_edges = (correlation_matrix > threshold).nonzero(as_tuple=False)  # Get indices of strong correlations
                
                # Step 3: Remove self-loops (no edges from a label to itself)
                edge_index = strong_edges[strong_edges[:, 0] != strong_edges[:, 1]].T  # Shape: [2, num_edges]
                label_dependency_gcn = LabelDependencyGCN(num_labels=xc.size(1), hidden_dim=64)
                label_dependency_gcn = label_dependency_gcn.to(device)
                xc, edge_index = xc.to(device), edge_index.to(device)
                xc = label_dependency_gcn(xc, edge_index)
                if xc is not None and xc.numel() > 0:  # Ensure xc is not empty
                    all_confidences.append(xc)
                
                out=xc
                #out_uncertain = out
                y_score = sigmoid(xc.data.cpu()/2)
                out_2=1-out
                out_temp=torch.reshape(out,(-1,1))
                out_2=torch.reshape(out_2,(-1,1))
                out_2D=torch.cat((out_temp,out_2),1)

            
                # Compute the risk labels
                risk_labels = risk_loss_criterion(test_pre_batch, rule_mus_batch, machine_mus_batch,
                                          rule_activate_batch, machine_activate_batch,
                                          machine_one_batch, y_score, labels=None)

                risk_labels = risk_labels.cuda()
        
                # Step 1: Update for Risky Labels Only (risk_labels == 1)
                mask_risky = (risk_labels == 0).long()
                
                if mask_risky.sum() > 0:  # Ensure there are risky labels to update
                    optimizer.zero_grad()  # Clear previous gradients
                    Loss_risky = LSLLoss(out, mask_risky) 
                    Loss_risky = Loss_risky.sum() / mask_risky.sum()  # Normalize loss
                    Loss_risky.backward(retain_graph=True)  # Backpropagate for risky labels, retain graph for uncertain
                    optimizer.step()  # Update model
                
                #epoch_loss += Loss_risky.item()

            # Step 2: Evaluate performance after risky label updates (before uncertain updates)
            #avg_confidence_high = torch.cat(all_confidences).mean().item()
            if num_batches_high > 0:
                avg_confidence_high = sum_confidence_high / num_batches_high
            else:
                avg_confidence_high = 0.0  # Fallback if no batches were processed
            print(f'Performance After High Risk Labels: {avg_confidence_high:.4f}')
            
            #test_acc, test_pre, test_f1_risky = ChexnetTrainer.test('/home/15t/Gul/Datasets/{}/test'.format(store_name), model,
            #                                             'RESNET-101', class_num,
            #                                             False, 1, 256, 244)
            
            # Save the model state after risky label updates
            model_risky_high = {k: v.clone() for k, v in model.state_dict().items()}
            
            # Step 3: Update for Uncertain Labels (risk_labels == -1)
            #if mask_uncertain.sum() > 0:  # Ensure there are uncertain labels to update
            model.train()
            for batch_idx, (inputs, targets, paths) in enumerate(TestDataLoader):
                if inputs.shape[0] < batch_size:
                    continue
        
                inputs, targets = inputs.to(device), targets.to(device)
                index = []

                # we just need class_name and image_name
                paths = list(paths)
                for path_i in range(len(paths)):
                    paths[path_i] = os.path.basename(
                        paths[path_i])
                    # print(paths[path_i])
                    #index.append(test_ids_dict.get(paths[path_i], -1))
                    index.append(test_ids_dict[paths[path_i]])
                #               print(index)

                test_pre_batch = test_pre[index]
                rule_mus_batch = rule_mus[index]
                machine_mus_batch = machine_mus[index]
                rule_activate_batch = rule_activate[index]
                machine_activate_batch = machine_activate[index]
                machine_one_batch = machine_one[index]    
                chex=1
                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                    
                optimizer.zero_grad()
                try:
                    x4, out = model(inputs)
    
                except:
                    out = model(inputs)
                batch_confidence_medium = out.mean().item()
    
                sum_confidence_medium += batch_confidence_medium
                num_batches_medium += 1
                
                correlation_matrix = torch.corrcoef(out.T)  # Correlation between labels

                # Step 2: Define a threshold for strong dependencies
                threshold = 0.5
                strong_edges = (correlation_matrix > threshold).nonzero(as_tuple=False)  # Get indices of strong correlations
                
                # Step 3: Remove self-loops (no edges from a label to itself)
                edge_index = strong_edges[strong_edges[:, 0] != strong_edges[:, 1]].T  # Shape: [2, num_edges]
                label_dependency_gcn = LabelDependencyGCN(num_labels=out.size(1), hidden_dim=64)
                label_dependency_gcn = label_dependency_gcn.to(device)
                out, edge_index = out.to(device), edge_index.to(device)
                out = label_dependency_gcn(out, edge_index)
                all_confidences.append(out)
                
                # Forward pass
                # Check if the model returns a tuple (e.g., (features, logits))
                if isinstance(out, tuple):
                    out = out[1]
                out=out.view(bs, n_crops, -1).mean(1)
                #out_uncertain = out
                y_score = sigmoid(xc.data.cpu()/2)
                #out = 
                # Compute the risk labels
                risk_labels = risk_loss_criterion(test_pre_batch, rule_mus_batch, machine_mus_batch,
                                              rule_activate_batch, machine_activate_batch,
                                              machine_one_batch, y_score, labels=None)
    
                risk_labels = risk_labels.cuda()
                # Update for Uncertain Labels (risk_labels == -1)
                mask_medium = (risk_labels == -1).long()
                if mask_medium.sum() > 0:
                    Loss_medium = LSLLoss1(out, mask_medium)
                    Loss_medium = Loss_medium.sum() / mask_medium.sum()  # Normalize loss
                    Loss_medium.backward()
                    optimizer.step()
            
                #epoch_loss += Loss_uncertain.item()
            
            # Step 4: Evaluate performance after uncertain updates
            if num_batches_medium > 0:
                avg_confidence_medium = sum_confidence_medium / num_batches_medium
            else:
                avg_confidence_medium = 0.0  # Fallback if no batches were process
            print(f'Performance After Medium Risk Labels: {avg_confidence_medium:.4f}')
            model_risky_medium = {k: v.clone() for k, v in model.state_dict().items()}
            
            model.train()
            for batch_idx, (inputs, targets, paths) in enumerate(TestDataLoader):
                if inputs.shape[0] < batch_size:
                    continue
        
                inputs, targets = inputs.to(device), targets.to(device)
                index = []

                # we just need class_name and image_name
                paths = list(paths)
                for path_i in range(len(paths)):
                    paths[path_i] = os.path.basename(
                        paths[path_i])
                    # print(paths[path_i])
                    #index.append(test_ids_dict.get(paths[path_i], -1))
                    index.append(test_ids_dict[paths[path_i]])
                #               print(index)

                test_pre_batch = test_pre[index]
                rule_mus_batch = rule_mus[index]
                machine_mus_batch = machine_mus[index]
                rule_activate_batch = rule_activate[index]
                machine_activate_batch = machine_activate[index]
                machine_one_batch = machine_one[index]    
                chex=1
                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                    
                optimizer.zero_grad()
                try:
                    x4, out = model(inputs)
    
                except:
                    out = model(inputs)
                
                correlation_matrix = torch.corrcoef(out.T)  # Correlation between labels

                # Step 2: Define a threshold for strong dependencies
                threshold = 0.5
                strong_edges = (correlation_matrix > threshold).nonzero(as_tuple=False)  # Get indices of strong correlations
                
                # Step 3: Remove self-loops (no edges from a label to itself)
                edge_index = strong_edges[strong_edges[:, 0] != strong_edges[:, 1]].T  # Shape: [2, num_edges]
                label_dependency_gcn = LabelDependencyGCN(num_labels=out.size(1), hidden_dim=64)
                label_dependency_gcn = label_dependency_gcn.to(device)
                out, edge_index = out.to(device), edge_index.to(device)
                out = label_dependency_gcn(out, edge_index)
                all_confidences.append(out)
                # Forward pass
                # Check if the model returns a tuple (e.g., (features, logits))
                if isinstance(out, tuple):
                    out = out[1]
                out=out.view(bs, n_crops, -1).mean(1)
                #out_uncertain = out
                y_score = sigmoid(xc.data.cpu()/2)
                #out = 
                # Compute the risk labels
                risk_labels = risk_loss_criterion(test_pre_batch, rule_mus_batch, machine_mus_batch,
                                              rule_activate_batch, machine_activate_batch,
                                              machine_one_batch, y_score, labels=None)
    
                risk_labels = risk_labels.cuda()
                # Update for Uncertain Labels (risk_labels == -1)
                mask_low = (risk_labels == 0).long()
                if mask_low.sum() > 0:
                    Loss_low = LSLLoss2(out, mask_low)
                    Loss_low = Loss_low.sum() / mask_low.sum()  # Normalize loss
                    Loss_low.backward()
                    optimizer.step()
            
                #epoch_loss += Loss_uncertain.item()
            
            # Step 4: Evaluate performance after uncertain updates
            avg_confidence_low = torch.cat(all_confidences).mean().item()
            print('Performance After Low Risk Labels: {avg_confidence_medium}')
            model_risky_low = {k: v.clone() for k, v in model.state_dict().items()}
            
            test_acc, test_pre, test_f1_uncertain = ChexnetTrainer.test('/home/15t/Gul/Datasets/{}/test'.format(store_name), model,
                                                         'RESNET-101', class_num,
                                                         False, 1, 256, 244)
            
            # Step 5: Compare the performance before and after uncertain updates based on confidence thresholds

            if avg_confidence_medium > avg_confidence_high:
                print(f"Model improved after medium-risk updates in epoch {epochID+1}")
            
                if avg_confidence_low > avg_confidence_medium:
                    print(f"Model improved further after low-risk updates in epoch {epochID+1}")
                    best_model_state = {k: v.clone() for k, v in model_risky_low.items()}
                    best_model_name = "model_low"
                    best_performance = avg_confidence_low
                else:
                    best_model_state = {k: v.clone() for k, v in model_risky_medium.items()}
                    best_model_name = "model_medium"
                    best_performance = avg_confidence_medium
            
            elif avg_confidence_high >= avg_confidence_medium:
                print(f"Keeping model after high-risk updates in epoch {epochID+1}")
                best_model_state = {k: v.clone() for k, v in model_risky_high.items()}
                best_model_name = "model_high"
                best_performance = avg_confidence_high
            
            else:
                # Revert model to its state before uncertain updates (risky stage)
                model.load_state_dict(model_risky_high)
                best_model_name = "model_high"
                print(f"Model reverted to its state after high-risk updates in epoch {epochID+1}")
            
            # Assign the best model state to the main model
            model.load_state_dict(best_model_state)
            print(f"Best model selected: {best_model_name}")
           
            # Step 6: Evaluate the final selected model on test data
            test_acc, test_pre, test_f1_final = Adaptive_Trainer.test('/home/15t/Gul/Datasets/{}/test'.format(store_name), model,
                                                                    'RESNET-101', class_num,
                                                                    False, 1, 256, 244)
            if test_f1_final > max_test_acc:
                max_test_acc = test_f1_final
                old_models = sorted(glob(join(exp_dir, 'max_*.pth')))
                
                # Remove the oldest saved model if it exists
                if len(old_models) > 0: 
                    os.remove(old_models[0])
                
                # Save the new best model based on test accuracy
                torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict()  # Ensure this is correctly placed if needed
                   },  # <- Closing parenthesis for the dictionary
                   os.path.join(exp_dir, "max_test_acc_{:.2f}.pth".format(max_test_acc)),
                   _use_new_zipfile_serialization=False)
            
                print(f"New best model saved with test accuracy: {max_test_acc:.2f}")

         
        if best_model_state:
            adaptive_lr_scheduler.step(test_f1)

        print(max_test_acc)





    # --------------------------------------------------------------------------------



    # ---- Test the trained network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- class_num - number of output classes
    # ---- batch_size - batch size
    # ---- nb_epoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the model_path file
    # ---- model_path - if not None loads the model and continues training

    def train_t(pathImgTest, pathModel, nnArchitecture, class_num, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):
        model = pathModel
        model.eval()
        model.cuda()

      
        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        chex=1
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(256))
        
        transformList.append(transforms.FiveCrop(224))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test = transforms.Compose(transformList)
       
        
        testset = CustomDataset(image_dir='/home/Gul/Datasets/BCNB/BCNB/WSIs/',
                              metadata_file='/home/Gul/Datasets/BCNB/BCNB/patient-clinical-data.xlsx',
                              ids_file='/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/train_id.txt',
                              transform=transform_test)
        
        
        
        testloader = torch.utils.data.DataLoader(
          testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        batch_hamming_losses = []
        distribution_x4 = []
        distribution_xc = []
        paths = []
        y_pred, y_true, y_score = [], [], []
        with torch.no_grad():

            for _, (inputs, targets, paths_batch) in enumerate(tqdm(testloader, ncols=80)):
                        if chex == 1:
                            if inputs.dim() == 4:
                                bs = inputs.size(0)
                                n_crops = 1
                                c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                                
                            elif inputs.dim() == 5:
                                bs, n_crops, c, h, w = inputs.size()
                                inputs = inputs.view(-1, c, h, w)  # Reshape to [batch_size * n_crops, c, h, w]
            
                            
            
                        inputs, targets = inputs.cuda(), targets.cuda()
                        
                        x4, xc = model(inputs)
            
                        if chex == 1: 
                          xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                          x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
                          
                        predicted = torch.sigmoid(xc.data) > 0.5
                        
                        predicted = predicted.cpu().numpy().astype(int)
                        
                        y_score.extend(sigmoid(xc.data.cpu()))
                        y_pred.extend(predicted.tolist())
                        y_true.extend(targets.cpu().numpy())
                        # Compute Hamming Loss
                        batch_hamming_loss = hamming_loss(y_true, y_pred)
                        batch_hamming_losses.append(batch_hamming_loss)
                        distribution_x4.extend(x4.cpu().tolist())
                        distribution_xc.extend(xc.cpu().tolist())
                        #y_true.extend(targets.cpu().tolist())
                        paths.extend(paths_batch)
                        y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                        varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                        y_score_t = torch.tensor(y_score_t)
                        varOutput_f = torch.tensor(varOutput_f)
                        varOutput_n = torch.reshape(y_score_t, (-1, 1))
                        varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                        varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                        y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)
                
            avg_hamming_loss = np.mean(batch_hamming_losses)
            std_hamming_loss = np.std(batch_hamming_losses)
            y_score = [tensor.tolist() for tensor in y_score]
            y_score = np.array(y_score)
            
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            distribution_xc_per_image = []
            num_labels = 3
            for i in range(len(distribution_xc)):
                image_predictions = []
                for j in range(num_labels):
                    
                    image_predictions.append(distribution_xc[i][j])
                    
                    image_predictions.append(1 - distribution_xc[i][j])
                distribution_xc_per_image.append(image_predictions)
            predictions_binary = (y_pred > 0.5).astype(int)
            true_labels_binary = y_true.astype(int)
            
            
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            print(f"predictions_binary shape: {predictions_binary.shape}")
            print(f"true_labels_binary shape: {true_labels_binary.shape}")

            train_accuracies = []
            train_precisions = []
            train_recalls = []
            train_f1_scores = []
            
            for label in range(num_labels):
                label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
                label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
                label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])
        
                train_accuracies.append(label_accuracy)
                train_precisions.append(label_precision)
                train_recalls.append(label_recall)
                train_f1_scores.append(label_f1_score)
        
            train_accuracy = 100.0 * np.mean(train_accuracies)
            train_precision = 100.0 *  np.mean(train_precisions)
            train_recall = 100.0 * np.mean(train_recalls)
            train_f1_score = 100.0 * np.mean(train_f1_scores)
        
            
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( train_accuracy, train_f1_score,
                                                                                  train_precision, train_recall, avg_hamming_loss
                                                                                  
                                                                                  ))
            
            y_score = torch.Tensor(distribution_xc_per_image)
            
            return train_accuracy,y_score

    def test(pathImgTest, pathModel, nnArchitecture, class_num, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):
       
        model = pathModel
       
        model.eval()
        model.cuda()

        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        chex=1
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(256))
        transformList.append(transforms.FiveCrop(224))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test = transforms.Compose(transformList)
        testset = CustomDataset(image_dir='/home/Gul/Datasets/BCNB/BCNB/WSIs/',
                              metadata_file='/home/Gul/Datasets/BCNB/BCNB/patient-clinical-data.xlsx',
                              ids_file='/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/test_id.txt',
                              transform=transform_test)
        testloader = torch.utils.data.DataLoader(
          testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        batch_hamming_losses =[]
        distribution_x4 = []
        distribution_xc = []
        paths = []
        y_pred, y_true, y_score, y_score_auc = [], [], [], []
        with torch.no_grad():

            for _, (inputs, targets, paths_batch) in enumerate(tqdm(testloader, ncols=80)):
                        if chex == 1:
                            if inputs.dim() == 4:
                                bs = inputs.size(0)
                                n_crops = 1
                                c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                                
                            elif inputs.dim() == 5:
                                bs, n_crops, c, h, w = inputs.size()
                                inputs = inputs.view(-1, c, h, w)  
            
                        inputs, targets = inputs.cuda(), targets.cuda()
                        x4, xc = model(inputs)
            
                        if chex == 1: 
                          xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                          x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
                          
                        predicted = torch.sigmoid(xc.data) > 0.5
                        predicted = predicted.cpu().numpy().astype(int)
                        y_score.extend(sigmoid(xc.data.cpu()))
                        y_score_auc.extend(torch.sigmoid(xc).cpu().numpy())
                        y_pred.extend(predicted.tolist())
                        y_true.extend(targets.cpu().numpy())
                        batch_hamming_loss = hamming_loss(y_true, y_pred)
                        batch_hamming_losses.append(batch_hamming_loss)
                        distribution_x4.extend(x4.cpu().tolist())
                        distribution_xc.extend(xc.cpu().tolist())
                        paths.extend(paths_batch)
                        y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                        varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                        y_score_t = torch.tensor(y_score_t)
                        varOutput_f = torch.tensor(varOutput_f)
                        varOutput_n = torch.reshape(y_score_t, (-1, 1))
                        varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                        varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                        y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)
                
            y_true_auc = np.array(y_true)
            y_score_auc = np.array(y_score_auc)
            print("This is true label:  ")
            print(y_true_auc)
            print("This is Predicted label:  ")
            print(y_score_auc)
            num_labels = 3
            avg_hamming_loss = np.mean(batch_hamming_losses)
            std_hamming_loss = np.std(batch_hamming_losses)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            distribution_xc_per_image = []
            num_labels = 3
            # Iterate over each sample and label index
            for i in range(len(distribution_xc)):
                image_predictions = []
                for j in range(num_labels):
                    # Append the raw output value for positive label
                    image_predictions.append(distribution_xc[i][j])
                    # Append the complement of raw output value for negative label
                    image_predictions.append(1 - distribution_xc[i][j])
                distribution_xc_per_image.append(image_predictions)
            predictions_binary = (y_pred > 0.5).astype(int)
            true_labels_binary = y_true.astype(int)
            
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            print(f"predictions_binary shape: {predictions_binary.shape}")
            print(f"true_labels_binary shape: {true_labels_binary.shape}")
        
            train_accuracies = []
            train_precisions = []
            train_recalls = []
            train_f1_scores = []
            train_aucs = []
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            for label in range(num_labels):
                label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
                label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
                label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_auc = roc_auc_score(y_true_auc[:, label], y_score_auc[:, label])
                label_auc1 = roc_auc_score(y_true_auc[:, label], y_score_auc[:, label], average='macro')
                label_auc2 = roc_auc_score(y_true_auc[:, label], y_score_auc[:, label], average='micro')
                label_auc3 = roc_auc_score(y_true_auc[:, label], y_score_auc[:, label], average='weighted')
                print('Macro')
                print(label_auc1)
                print('Micro')
                print(label_auc2)
                print('weighted')
                print(label_auc3)
                
                print("Dataset Label \t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                    label, label_accuracy, label_f1_score, label_precision, label_recall, label_auc, avg_hamming_loss, std_hamming_loss))

                train_accuracies.append(label_accuracy)
                train_precisions.append(label_precision)
                train_recalls.append(label_recall)
                train_f1_scores.append(label_f1_score)
                train_aucs.append(label_auc)
            train_accuracy = 100.0 * np.mean(train_accuracies)
            train_precision = 100.0 * np.mean(train_precisions)
            train_recall = 100.0 * np.mean(train_recalls)
            train_f1_score = 100.0 * np.mean(train_f1_scores)
            train_auc = 100.0 * np.mean(train_aucs)
            confusion_matrices = multilabel_confusion_matrix(true_labels_binary, predictions_binary)
            print("Dataset over all \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\t{:.2f}\n".format( train_accuracy, train_f1_score,
                                                                                  train_precision, train_recall, train_auc, 
                                                                                  avg_hamming_loss
                                                                                  ))
          
            
            y_score = torch.Tensor(distribution_xc_per_image)
            return train_accuracy, y_score, train_f1_score
            
            
    def Valtest(pathImgTest, pathModel, nnArchitecture, testdataloader, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):

  
        model = pathModel
 
        model.eval()
        model.cuda()

        
        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.RandomResizedCrop(224))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transform_test = transforms.Compose(transformList)
        chex=0

        with torch.no_grad():

            testset = CustomDataset(image_dir='/home/Gul/Datasets/BCNB/BCNB/WSIs/',
                              metadata_file='/home/Gul/Datasets/BCNB/BCNB/patient-clinical-data.xlsx',
                              ids_file='/home/Gul/Datasets/BCNB/BCNB/dataset-splitting/val_id.txt',
                              transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            total_hamming_loss = 0
            distribution_x4 = []
            distribution_xc = []
            paths = []
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, paths_batch) in enumerate(tqdm(testloader, ncols=80)):
                        if chex == 1:
                            if inputs.dim() == 4:
                                bs = inputs.size(0)
                                n_crops = 1
                                c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                                
                            elif inputs.dim() == 5:
                                bs, n_crops, c, h, w = inputs.size()
                                inputs = inputs.view(-1, c, h, w)  # Reshape to [batch_size * n_crops, c, h, w]

            
                        inputs, targets = inputs.cuda(), targets.cuda()
                        x4, xc = model(inputs)
            
                        if chex == 1: 
                          xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                          x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
                          
                        predicted = torch.sigmoid(xc.data) > 0.5
                        
                        predicted = predicted.cpu().numpy().astype(int)
                        
                        y_score.extend(sigmoid(xc.data.cpu()))
                        y_pred.extend(predicted.tolist())
                        y_true.extend(targets.cpu().numpy())
                        batch_hamming_loss = hamming_loss(y_true, y_pred)
                        total_hamming_loss += batch_hamming_loss
                        distribution_x4.extend(x4.cpu().tolist())
                        distribution_xc.extend(xc.cpu().tolist())
                        
                        paths.extend(paths_batch)

                        y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                        varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                        y_score_t = torch.tensor(y_score_t)
                        varOutput_f = torch.tensor(varOutput_f)
                        
                        varOutput_n = torch.reshape(y_score_t, (-1, 1))
                        varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                       
                        
                        varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                        y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)

                
            avg_hamming_loss = total_hamming_loss / len(testloader)
            y_score = [tensor.tolist() for tensor in y_score]
            y_score = np.array(y_score)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_score = np.concatenate(y_score, axis=0)
            distribution_xc_per_image = []
            num_labels = 3
            # Iterate over each sample and label index
            for i in range(len(distribution_xc)):
                image_predictions = []
                for j in range(num_labels):
                    # Append the raw output value for positive label
                    image_predictions.append(distribution_xc[i][j])
                    # Append the complement of raw output value for negative label
                    image_predictions.append(1 - distribution_xc[i][j])
                distribution_xc_per_image.append(image_predictions)
             # Evaluate predictions
            predictions_binary = (y_pred > 0.5).astype(int)
            true_labels_binary = y_true.astype(int)
            num_labels = 3
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            print(f"predictions_binary shape: {predictions_binary.shape}")
            print(f"true_labels_binary shape: {true_labels_binary.shape}")
        
            train_accuracies = []
            train_precisions = []
            train_recalls = []
            train_f1_scores = []
            predictions_binary = predictions_binary.reshape(-1, num_labels)
            true_labels_binary = true_labels_binary.reshape(-1, num_labels)
            for label in range(num_labels):
                label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
                label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
                label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])
        
                train_accuracies.append(label_accuracy)
                train_precisions.append(label_precision)
                train_recalls.append(label_recall)
                train_f1_scores.append(label_f1_score)
        
            train_accuracy = 100.0 * np.mean(train_accuracies)
            train_precision = 100.0 * np.mean(train_precisions)
            train_recall = 100.0 * np.mean(train_recalls)
            train_f1_score = 100.0 * np.mean(train_f1_scores)
        
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( train_accuracy, train_f1_score,
                                                                                  train_precision, train_recall, avg_hamming_loss
                                                                                  
                                                                                  ))
          
            y_score = torch.Tensor(distribution_xc_per_image)
            
            return train_accuracy,y_score







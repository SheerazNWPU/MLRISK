
import os, pickle, time, shutil
from os.path import join
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from Densenet import densenet121, densenet161, densenet169, densenet201
from Folder import ImageFolder
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from Vgg import vgg11, vgg13, vgg16, vgg19
from torchvision.models import alexnet
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
from scipy.special import softmax
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
begin = time.time()
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cnn',  default='r50', help='dataset dir')
parser.add_argument('-d','--dir',  default='', help='dataset dir')  # chest_xray, hosp
parser.add_argument('-s','--save_dir',  default='Multilabel50', help='save_dir')
parser.add_argument('-m','--multiple', default=2, type=int, help='multiple of input size')
parser.add_argument('-g','--gpu',  default='1', help='set 0,1 to use multi gpu for example')
args = parser.parse_args()


# exp settings
cnn = args.cnn
datasets_dir = args.dir
exp_dir = "/YourSavePath/result_archive/{}".format(args.save_dir)
batch_size = 1
# os.makedirs(exp_dir, exist_ok=True)
# shutil.copy('get_distribution.py', join(exp_dir, 'get_distribution.py'))


# data settings
data_dir = join("/home/4t/SG/", datasets_dir)
#data_sets = ['train', 'val', 'test']
#data_sets = ['Validation','Test']
# nb_class = len(os.listdir(join(data_dir, data_sets[0])))
nb_class = 3
re_size = int(128 * args.multiple)
crop_size = 112 * args.multiple
chex=1

## Allow Large Images
Image.MAX_IMAGE_PIXELS = None

# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


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


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.RandomResizedCrop(224))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)
train_transform = transforms.Compose(transformList)





normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.Resize(256))

transformList.append(transforms.FiveCrop(224))
transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
transform_test = transforms.Compose(transformList)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create datasets and dataloaders
train_dataset = CustomDataset(image_dir='/DatasetPath/BCNB/WSIs/',
                               metadata_file='/DatasetPath/BCNB/patient-clinical-data.xlsx',
                               ids_file='/DatasetPath/BCNB/dataset-splitting/train_id.txt',
                               transform=train_transform)

val_dataset = CustomDataset(image_dir='/DatasetPath/BCNB/WSIs/',
                             metadata_file='/DatasetPath/BCNB/patient-clinical-data.xlsx',
                             ids_file='/DatasetPath/BCNB/dataset-splitting/val_id.txt',
                             transform=transform_test)

test_dataset = CustomDataset(image_dir='/DatasetPath/BCNB/WSIs/',
                              metadata_file='/DatasetPath/BCNB/patient-clinical-data.xlsx',
                              ids_file='/DatasetPath/BCNB/dataset-splitting/test_id.txt',
                              transform=transform_test)

data_sets = [train_dataset, val_dataset, test_dataset]
# Net settings
print('===== {} ====='.format(args.save_dir))
model_zoo = {'r18':resnet18, 'r34':resnet34, 'r50':resnet50, 'r101':resnet101, 'r152':resnet152, 'd121':densenet121, 'd161':densenet161, 'd169':densenet169, 'd201':densenet201,
             'eb0':efficientnet_b0, 'eb1':efficientnet_b1, 'eb2':efficientnet_b2, 'eb3':efficientnet_b3,
             'eb4':efficientnet_b4, 'eb5':efficientnet_b5, 'eb6':efficientnet_b6,  'eb7':efficientnet_b7,
             'rx50':resnext50_32x4d, 'alex':alexnet, 'wrn50':wide_resnet50_2, 'wrn101':wide_resnet101_2,
             'v11':vgg11, 'v13':vgg13, 'v16':vgg16, 'v19':vgg19}
net = model_zoo[cnn](pretrained=False)

if cnn.startswith("r"):
    net.fc = nn.Linear(net.fc.in_features, nb_class)
    # net.fc = nn.Sequential(nn.Dropout(p=args.drop_out), nn.Linear(net.fc.in_features, nb_class))  # for resnet
elif cnn.startswith('w'):
    net.fc = nn.Linear(net.fc.in_features, nb_class)
elif cnn.startswith('v'):
    net.classifier = nn.Linear(net.classifier[0].in_features, nb_class) # for VGG
elif cnn.startswith('d'):
    net.classifier = nn.Linear(net.classifier.in_features, nb_class)
    # net.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(net.classifier.in_features, nb_class))  # for densenet
elif cnn.startswith('e'):
    net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
net.cuda()


net_file_name = '/YourSavePath/result_archive/Multilabel50/max_f1_0.56.pth'
try:
        try: net.load_state_dict(torch.load(net_file_name)['model'])
        except: net.load_state_dict(torch.load(net_file_name))
except:
        model = torch.nn.DataParallel(net)
        try: model.load_state_dict(torch.load(net_file_name)['model'])
        except: model.load_state_dict(torch.load(net_file_name))

net.cuda()
net.eval()

scores = []

for data_set in data_sets:
    #print(len(data_set))
    testloader = torch.utils.data.DataLoader(
        data_set, batch_size=1, shuffle=False, num_workers=0
    )
    #for batch in testloader:
   #   print(batch)
    #  #break
    distribution_x4 = []
    distribution_xc = []
    y_pred, y_true, y_score = [], [], []
    paths = []
    test_loss = correct = total = 0

    with torch.no_grad():
       for _, (inputs, targets, paths_batch) in enumerate(tqdm(testloader, ncols=80)):
            if chex == 1:
                if inputs.dim() == 4:
                    bs = inputs.size(0)
                    n_crops = 1
                    c, h, w = inputs.size(1), inputs.size(2), inputs.size(3)
                    
                elif inputs.dim() == 5:
                    bs, n_crops, c, h, w = inputs.size()
                #bs, n_crops, c, h, w = inputs.size()
                #inputs = inputs.view(-1, c, h, w)

            inputs, targets = inputs.cuda(), targets.cuda()
            #print(model(inputs))
            x4, xc = net(inputs)

            if chex == 1: 
              xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
              x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)
              
            predicted = torch.sigmoid(xc.data) > 0.5
            print(predicted)
            predicted = predicted.cpu().numpy().astype(int)
            
            y_score.extend(softmax(xc.data.cpu().tolist(), axis=1))
            y_pred.extend(predicted.tolist())
            y_true.extend(targets.cpu().numpy())

            distribution_x4.extend(x4.cpu().tolist())
            distribution_xc.extend(xc.cpu().tolist())
            #y_true.extend(targets.cpu().tolist())
            paths.extend(paths_batch)
    predictions = np.array(distribution_xc)
    true_labels = np.array(y_true)
    
    
    predictions_binary = (predictions > 0.5).astype(int)
    true_labels_binary = true_labels.astype(int)
    
    print(f"predictions_binary shape: {predictions_binary.shape}")
    print(f"true_labels_binary shape: {true_labels_binary.shape}")
    

    num_labels = predictions_binary.shape[1]
    predictions_per_label = [predictions_binary[:, i] for i in range(num_labels)]

    # Split data into separate lists for each label
    y_true_per_label = [[] for _ in range(num_labels)]
    
    distribution_xc_per_image = []
    
    # Iterate over each sample and label index
    for i in range(len(distribution_xc)):
        image_predictions = []
        for j in range(num_labels):
            # Append the raw output value for positive label
            image_predictions.append(distribution_xc[i][j])
            # Append the complement of raw output value for negative label
            image_predictions.append(1 - distribution_xc[i][j])
        distribution_xc_per_image.append(image_predictions)



    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    for label in range(num_labels):
        label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
        label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0, average = 'macro')
        label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label], average = 'macro')
        label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label], average = 'macro')
        print("Label {}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t".format(label, label_accuracy, label_f1_score, label_precision, label_recall))
        test_accuracies.append(label_accuracy)
        test_precisions.append(label_precision)
        test_recalls.append(label_recall)
        test_f1_scores.append(label_f1_score)
    
    test_accuracy = np.mean(test_accuracies)
    test_precision = np.mean(test_precisions)
    test_recall = np.mean(test_recalls)
    test_f1_score = np.mean(test_f1_scores)
    
    print("Dataset {}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t".format('Test Set', test_accuracy, test_f1_score, test_precision, test_recall))
    # # === 保存 pkl===
    with open(os.path.join(exp_dir, "{}_{:.2f}\n".format(data_set, test_accuracy)), "a+") as file: pass
    pickle.dump(y_true, open(join(exp_dir, "targets_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(y_pred, open(join(exp_dir, "predictions_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(paths, open(join(exp_dir, "paths_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(distribution_x4, open(join(exp_dir, "distribution_x4_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(distribution_xc, open(join(exp_dir, "distribution_xc_{}.pkl".format(data_set)), 'wb+'))
   
    
    if data_set == train_dataset:
        pd.DataFrame(y_true).to_csv(join(exp_dir, "targets_{}.csv".format('train')), index=None, header=None)
        pd.DataFrame(y_pred).to_csv(join(exp_dir, "predictions_{}.csv".format('train')), index=None, header=None)
        #header = [f'negative_class_label_{i}', f'positive_class_label_{i}' for i in range(num_labels)]
        pd.DataFrame(distribution_xc_per_image).to_csv(join(exp_dir, "distribution_xc_{}.csv".format('train')), index=None, header=None)
        pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format('train')), index=None, header=None)
        pd.DataFrame(paths).to_csv(join(exp_dir, "paths_{}.csv".format('train')), index=None, header=None)
        #pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format('train')), index=None, header=None)
        #pd.DataFrame(distribution_xc).to_csv(join(exp_dir, "distribution_xc_{}.csv".format('train')), index=None, header=None)
    elif data_set == val_dataset:
        pd.DataFrame(y_true).to_csv(join(exp_dir, "targets_{}.csv".format('val')), index=None, header=None)
        pd.DataFrame(y_pred).to_csv(join(exp_dir, "predictions_{}.csv".format('val')), index=None, header=None)
        #header = [f'negative_class_label_{i}', f'positive_class_label_{i}' for i in range(num_labels)]
        pd.DataFrame(distribution_xc_per_image).to_csv(join(exp_dir, "distribution_xc_{}.csv".format('val')), index=None, header=None)
        pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format('val')), index=None, header=None)

        pd.DataFrame(paths).to_csv(join(exp_dir, "paths_{}.csv".format('val')), index=None, header=None)
        #pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format('val')), index=None, header=None)
        #pd.DataFrame(distribution_xc).to_csv(join(exp_dir, "distribution_xc_{}.csv".format('val')), index=None, header=None)
    elif data_set == test_dataset:
        pd.DataFrame(y_true).to_csv(join(exp_dir, "targets_{}.csv".format('test')), index=None, header=None)
        pd.DataFrame(y_pred).to_csv(join(exp_dir, "predictions_{}.csv".format('test')), index=None, header=None)
        #header = [f'negative_class_label_{i}', f'positive_class_label_{i}' for i in range(num_labels)]
        pd.DataFrame(distribution_xc_per_image).to_csv(join(exp_dir, "distribution_xc_{}.csv".format('test')), index=None, header=None)
        pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format('test')), index=None, header=None)
        pd.DataFrame(paths).to_csv(join(exp_dir, "paths_{}.csv".format('test')), index=None, header=None)
        #pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format('test')), index=None, header=None)
        #pd.DataFrame(distribution_xc).to_csv(join(exp_dir, "distribution_xc_{}.csv".format('test')), index=None, header=None)
    

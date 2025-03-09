import os, pickle
import argparse
import random
import shutil
from os.path import join
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from Densenet import densenet121, densenet161, densenet169, densenet201
from Folder import ImageFolder
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from Vgg import vgg11, vgg13, vgg16, vgg19
from torchvision.models import (
    vit_b_16, vit_b_32, vit_l_16, vit_l_32  # ViT model variants
)

from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, global_mean_pool
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
#from Focal_loss import focal_loss
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append("..")
#from roc_auc_score_multiclass import roc_auc_score_multiclass

#######################
##### 1 - Setting #####
#######################
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-m','--multiple', default=2, type=int, help='multiple of input size')
parser.add_argument('--ckpt', default='', help='path of check_point model')
parser.add_argument('--requires_grad', default='all', help='the layers need finetune')
parser.add_argument('-d','--data_dir',  default='', help='dataset dir')
parser.add_argument('-c','--cnn',  default='r50', help='CNN model')
parser.add_argument('-b','--batch_size',  default=16, type=int, help='batch_size')
parser.add_argument('--wt',  default=0, type=int, help='weight loss')
parser.add_argument('-g','--gpu',  default='0', help='gpu id')
parser.add_argument('--train_set', default='train', help='name of training set')
parser.add_argument('--test_set', default='val', help='name of testing set')
parser.add_argument('-w','--num_workers',  default=0, type=int, help='num_workers')
parser.add_argument('-e','--epoch',  default=150, type=int, help='epoch')
parser.add_argument('--chex',  default=1, type=int, help='use chexnet setting or not')
parser.add_argument('-r','--random_seed',  default=0, type=int, help='random seed')
parser.add_argument('-s','--save_dir',  default='Multilabel50', help='save_dir')
parser.add_argument('-l','--label_smooth',  default=0, type=float, help='label_smooth')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning_rate')
parser.add_argument('--scheduler', default=0, type=int, help='use scheduler')
parser.add_argument('-v','--evaluate', default=1, type=int, help='test every epoch')
parser.add_argument('-a', '--amp', default=2, type=int, help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp')
args = parser.parse_args()

chex = args.chex  # use chexnet setting
num_epoch = args.epoch
begin_epoch = 1
seed = args.random_seed
cnn = args.cnn
batch_size = args.batch_size
if args.learning_rate: lr_begin = args.learning_rate
else: lr_begin = (batch_size / 256) * 0.1
use_amp = args.amp
opt_level = "O1"
test_every_epoch = args.evaluate

## Allow Large Images
Image.MAX_IMAGE_PIXELS = None

# data settings
#data_dir = join("/home/4t/SG/", args.data_dir)
# data_sets = ["hosp_val", 'hosp_test']
# data_sets = ["hosp_test", 'hosp_val']
data_sets = [args.train_set, args.test_set]
nb_class = 3
re_size = int(128 * args.multiple)
crop_size = 112 * args.multiple
exp_dir = "/YourSavePath/result_archive/{}".format(args.save_dir)
summaryWriter = SummaryWriter(exp_dir)


# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# Random seed setting
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # multi gpu
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
class ViTWithFeatures(nn.Module):
    def __init__(self, vit_model):
        super(ViTWithFeatures, self).__init__()
        self.vit = vit_model
        
    def forward(self, x):
        # Extract features from the transformer backbone (before the final classifier)
        features = self.vit.forward_features(x)  # Intermediate features (X4)
        logits = self.vit.head(features)  # Final logits (XC)
        return features, logits
# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dir, metadata_file, ids_file, transform=None):
        self.metadata = pd.read_excel(metadata_file)
        self.image_ids = np.loadtxt(ids_file, dtype=str)
        self.image_dir = image_dir
        self.transform = transform
        # Manually define the mapping for each categorical label
        self.er_mapping = {'Positive': 1, 'Negative': 0}
        self.pr_mapping = {'Positive': 1, 'Negative': 0}
        self.her2_mapping = {'Positive': 1, 'Negative': 0}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.image_ids[idx] + ".jpg")
        image = Image.open(image_name).convert('RGB')
    
        if self.transform:
            image = self.transform(image)
    
        label_er = self.metadata['ER'].map({'Positive': 1, 'Negative': 0}).values[idx]
        label_pr = self.metadata['PR'].map({'Positive': 1, 'Negative': 0}).values[idx]
        label_her2 = self.metadata['HER2'].map({'Positive': 1, 'Negative': 0}).values[idx]
    
        label = [label_er, label_pr, label_her2]
    
        return image, torch.FloatTensor(label)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataloader
# hosp_val	([0.478, 0.478, 0.478], [0.276, 0.276, 0.276])

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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(args.num_workers), drop_last=True
    
)


# Net settings
model_zoo = {
    'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
    'd121': densenet121, 'd161': densenet161, 'd169': densenet169, 'd201': densenet201,
    'v11': vgg11, 'v13': vgg13, 'v16': vgg16, 'v19': vgg19,
    'eb0': efficientnet_b0, 'eb1': efficientnet_b1, 'eb2': efficientnet_b2, 'eb3': efficientnet_b3,
    'eb4': efficientnet_b4, 'eb5': efficientnet_b5, 'eb6': efficientnet_b6, 'eb7': efficientnet_b7,
    'rx50': resnext50_32x4d, 'wrn50': wide_resnet50_2, 'wrn101': wide_resnet101_2,
    'vit_b_16': vit_b_16, 'vit_b_32': vit_b_32, 'vit_l_16': vit_l_16, 'vit_l_32': vit_l_32,  # ViT models
    'gcn': 'GCN', 'sage': 'GraphSAGE', 'gat': 'GAT'  # GNN models (names are placeholders)
}

# GNN model definition
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)
        
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = GraphSAGE(in_channels, 64)
        self.conv2 = GraphSAGE(64, out_channels)
    
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 64, heads=4, concat=True)
        self.conv2 = GATConv(64 * 4, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

net = model_zoo[cnn](pretrained=True)

if cnn == 'gcn':
    # If we are using a GNN model, define the input channels and output classes
    net = GCN(in_channels=256, out_channels=nb_class)  # Example, adjust in_channels based on your data
elif cnn == 'sage':
    net = GraphSAGE(in_channels=256, out_channels=nb_class)  # Adjust based on data
elif cnn == 'gat':
    net = GAT(in_channels=256, out_channels=nb_class)  # Adjust based on data
else:
    # For CNNs and ViTs
    net = model_zoo[cnn](pretrained=True)
    if cnn.startswith("r"):  # ResNet models
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif cnn.startswith('w'):  # Wide ResNet models
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif cnn.startswith('vgg'):  # VGG models
        net.classifier = nn.Linear(net.classifier[0].in_features, nb_class)
    elif cnn.startswith('d'):  # DenseNet models
        net.classifier = nn.Linear(net.classifier.in_features, nb_class)
    elif cnn.startswith('e'):  # EfficientNet models
        net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
    elif cnn.startswith('vit'):  # Vision Transformer models
        net.heads.head = nn.Linear(net.heads.head.in_features, nb_class)

# Move model to GPU
net.cuda()


# optimizer setting
# train_Loss = focal_loss(alpha=0.25)
if args.wt == 0: train_Loss = nn.BCELoss()  # , weight=torch.Tensor([0.19476485210143069, 0.09738242605071534, 0.1787430647820328, 0.11142796826956852, 0.17966680155093218, 0.10455797323339963, 0.13345691401192086]).cuda()
else: train_Loss = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

if args.ckpt:
    print('=== use ckpt.pth ===')
    ckpt = torch.load(args.ckpt)
    net.load_state_dict(ckpt['model'])
  


if args.requires_grad == 'fc':
    for name, param in net.named_parameters():
        if "fc" in name: param.requires_grad = True
        else: param.requires_grad = False
elif args.requires_grad == 'all':
    for name, param in net.named_parameters():
        param.requires_grad = True


# Training
os.makedirs(exp_dir, exist_ok=True)
shutil.copyfile("train.sh", exp_dir + "/train.sh")
shutil.copyfile("train.py", exp_dir + "/train.py")
shutil.copyfile("Folder.py", exp_dir + "/Folder.py")
shutil.copyfile("Densenet.py", exp_dir + "/Densenet.py")
shutil.copyfile("Resnet.py", exp_dir + "/Resnet.py")


# Amp
if use_amp == 1:  # use nvidia apex.amp
    print('\n===== Using NVIDIA AMP =====')
    from apex import amp

    # net.cuda()
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using NVIDIA AMP =====\n')
elif use_amp == 2:  # use torch.cuda.amp
    print('\n===== Using Torch AMP =====')
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using Torch AMP =====\n')

if len(args.gpu) > 1: net = torch.nn.DataParallel(net)


########################
##### 2 - Training #####
########################
min_train_loss = float('inf')
max_val_f1 = 0

for epoch in range(begin_epoch, num_epoch + 1):
    print("\n===== Epoch: {} / {} =====".format(epoch, num_epoch))
    net.train()
    lr_now = optimizer.param_groups[0]["lr"]
    train_loss = 0
    y_pred, y_true, y_score = [], [], []

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
        optimizer.zero_grad()
        y_true.append(targets.cpu().numpy())
        #inputs = inputs.half()
        inputs, targets = inputs.cuda(), targets.cuda()

        if use_amp == 1:  # use nvidia apex.amp
            #x4, xc = net(inputs)
            xc = net(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xc, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:  # use torch.cuda.amp
            with autocast():
                #x4, xc = net(inputs)
                xc = net(inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(xc, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #x4, xc = net(inputs)
            xc = net(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xc, targets)
            loss.backward()
            optimizer.step()

        #_, predicted = torch.max(xc.data, 1)
        y_score.append(softmax(xc.data.cpu().numpy(), axis=1))
        y_pred.append(xc.detach().cpu().numpy())
        train_loss += loss.item()

    if args.scheduler == 1:
        scheduler.step()

    train_loss /= len(train_loader)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    # Evaluate predictions
    predictions_binary = (y_pred > 0.5).astype(int)
    true_labels_binary = y_true.astype(int)

    print(f"predictions_binary shape: {predictions_binary.shape}")
    print(f"true_labels_binary shape: {true_labels_binary.shape}")
    #print(predictions_binary.shape[1])
    num_labels = true_labels_binary.shape[1]

    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1_scores = []
    #predictions_binary = predictions_binary.reshape(-1, num_labels)
    #true_labels_binary = true_labels_binary.reshape(-1, num_labels)
    for label in range(num_labels):
        label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
        label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
        label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
        label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])

        train_accuracies.append(label_accuracy)
        train_precisions.append(label_precision)
        train_recalls.append(label_recall)
        train_f1_scores.append(label_f1_score)

    train_accuracy = np.mean(train_accuracies)
    train_precision = np.mean(train_precisions)
    train_recall = np.mean(train_recalls)
    train_f1_score = np.mean(train_f1_scores)

    print(
        "Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f} | F1: {:.3f}".format(
            lr_now, train_loss, train_accuracy, train_f1_score
        )
    )

    # Evaluating
    if test_every_epoch == 0:
        # Save last epoch model
        torch.save({'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    },
                   os.path.join(exp_dir, "ckpt.pth"),
                   _use_new_zipfile_serialization=False)
    else:
        net.eval()
        val_Loss = nn.BCELoss()
        val_loss = 0
        with torch.no_grad():
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets) in enumerate(tqdm(val_loader, ncols=80)):
                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)
                    
                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    xc = net(inputs)
                except:
                    xc = net(inputs)

                if chex == 1:
                    xc = xc.squeeze().view(bs, n_crops, -1).mean(1)

                #_, predicted = torch.max(xc.data, 1)
                y_score.append(softmax(xc.data.cpu().numpy(), axis=1))
                y_pred.append(xc.detach().cpu().numpy())
                y_true.append(targets.cpu().numpy())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(xc, targets)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            y_score = np.concatenate(y_score, axis=0)

            predictions_binary = (y_pred > 0.5).astype(int)
            true_labels_binary = y_true.astype(int)
            val_accuracies = []
            val_precisions = []
            val_recalls = []
            val_f1_scores = []

            for label in range(num_labels):
                label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
                label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
                label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
                label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])

                val_accuracies.append(label_accuracy)
                val_precisions.append(label_precision)
                val_recalls.append(label_recall)
                val_f1_scores.append(label_f1_score)

            val_accuracy = np.mean(val_accuracies)
            val_precision = np.mean(val_precisions)
            val_recall = np.mean(val_recalls)
            val_f1_score = np.mean(val_f1_scores)

            print('{} | Loss: {:.4f} | Acc: {:.3f} | F1: {:.3f} | '.format(data_sets[-1], val_loss, val_accuracy, val_f1_score))
            summaryWriter.add_scalars(main_tag='', tag_scalar_dict={"train_loss":train_loss,
                                                                    'train_recall': train_recall,
                                                                    'train_precision': train_precision,
                                                                    "train_f1":train_f1_score,
                                                                    "train_acc": train_accuracy,
                                                                    #"train_auc": train_auc,
                                                                    "val_loss":val_loss,
                                                                    'val_recall':val_recall,
                                                                    'val_precision':val_precision,
                                                                    "val_f1":val_f1_score,
                                                                    "val_acc": val_accuracy,
                                                                    #"val_auc": val_auc,
                                                                    'lr':lr_now}, global_step=epoch)
            patience = 10  # Number of epochs without improvement
            epochs_since_improvement = 0
            if val_f1_score > max_val_f1 and epoch > 40 :
                max_val_f1 = val_f1_score
                epochs_since_improvement = 0
                old_models = sorted(glob(join(exp_dir, 'max_*.pth')))
                if len(old_models) > 0: os.remove(old_models[0])
                torch.save({'epoch': epoch,
                            'model': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            },
                           os.path.join(exp_dir, "max_f1_{:.2f}.pth".format(max_val_f1)),
                           _use_new_zipfile_serialization=False)
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f'Early stopping: No improvement in {patience} epochs.')
                break



########################
##### 3 - Testing  #####
########################
print("\n\n===== TESTING =====")
print(args.save_dir)
# reload the model for testing
net = model_zoo[cnn]()

if cnn.startswith("r"):
    net.fc = nn.Linear(net.fc.in_features, nb_class)
elif cnn.startswith('w'):
    net.fc = nn.Linear(net.fc.in_features, nb_class)
elif cnn.startswith('v'):
    net.classifier = nn.Linear(net.classifier[0].in_features, nb_class)
elif cnn.startswith('d'):
    net.classifier = nn.Linear(net.classifier.in_features, nb_class)
elif cnn.startswith('e'):
    net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)

if len(args.gpu) > 1:
    net = torch.nn.DataParallel(net)

net_file_name = glob(join(exp_dir, "max_*.pth"))[0]
net.load_state_dict(torch.load(join(exp_dir, net_file_name))['model'])
net.eval()
net.cuda()

y_pred, y_true, y_score = [], [], []
y_pred = []
y_true = []
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
for _, (inputs, targets) in enumerate(tqdm(testloader, ncols=80)):
    if chex == 1:
        bs, n_crops, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)
        #bs, n_crops, c, h, w = inputs.size()
        #inputs = inputs.view(-1, c, h, w)
    #inputs = inputs.half()
    inputs, targets = inputs.cuda(), targets.cuda()
    try:
        #x4, xc = net(inputs)
        xc = net(inputs)
    except:
        xc = net(inputs)

    if chex == 1:
        xc = xc.squeeze().view(bs, n_crops, -1).mean(1)

    #_, predicted = torch.max(xc.data, 1)
    y_pred.append(xc.detach().cpu().numpy())
    y_true.append(targets.cpu().numpy())

y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)
y_score = np.concatenate(y_score, axis=0)

predictions_binary = (y_pred > 0.5).astype(int)
true_labels_binary = y_true.astype(int)

print(f"predictions_binary shape: {predictions_binary.shape}")
print(f"true_labels_binary shape: {true_labels_binary.shape}")

num_labels = true_labels_binary.shape[1]
#num_labels = predictions_binary.shape[1]

test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []
#predictions_binary = predictions_binary.reshape(-1, num_labels)
#true_labels_binary = true_labels_binary.reshape(-1, num_labels)
for label in range(num_labels):
    label_accuracy = np.mean(predictions_binary[:, label] == true_labels_binary[:, label])
    label_precision = precision_score(true_labels_binary[:, label], predictions_binary[:, label], zero_division=0)
    label_recall = recall_score(true_labels_binary[:, label], predictions_binary[:, label])
    label_f1_score = f1_score(true_labels_binary[:, label], predictions_binary[:, label])

    test_accuracies.append(label_accuracy)
    test_precisions.append(label_precision)
    test_recalls.append(label_recall)
    test_f1_scores.append(label_f1_score)

test_accuracy = np.mean(test_accuracies)
test_precision = np.mean(test_precisions)
test_recall = np.mean(test_recalls)
test_f1_score = np.mean(test_f1_scores)

print("Dataset {}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t".format('Test Set', test_accuracy, test_f1_score, test_precision, test_recall))

with open(os.path.join(exp_dir, "{:.2f}_{}.txt".format(test_f1_score, 'test')), "a+") as file:
    pass
    #Logging
    with open(os.path.join(exp_dir, "{:.2f}_{}\n".format(test_f1_score, data_set)), "a+") as file:
        pass
    pickle.dump(y_true, open(join(exp_dir, "targets_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(y_pred, open(join(exp_dir, "predictions_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(paths, open(join(exp_dir, "paths_{}.pkl".format(data_set)), 'wb+'))
    #pickle.dump(distribution_x4, open(join(exp_dir, "distribution_x4_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(distribution_xc, open(join(exp_dir, "distribution_xc_{}.pkl".format(data_set)), 'wb+'))
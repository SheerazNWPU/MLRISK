import os
import numpy as np
import time
import sys

from Adaptive_Training import Adaptive_Trainer


#--------------------------------------------------------------------------------   

def runTrain():
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RSENET101='RESNET-101'
    RSENET50='RESNET-50'
    WIDERSENET50='WIDE-RESNET-50'
    EFFICIENTNETB4 = 'EFFICIENT-NET_B4'
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # ---- Path to the directory with images
    pathImgTrain = ''
    pathImgVal = ''
    pathImgTest = ''

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    nnArchitecture = RSENET50
    nnIsTrained = True
    nnClassCount = 3  # 14

    # ---- Training settings: batch size, maximum number of epochs
    trBatchSize = 8
    trMaxEpoch =20

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'

   
    print('=== Training NN architecture = ', nnArchitecture, '===')
    Adaptive_Trainer.train(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize,
                        trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, 10, 'MultilabelP', '/home/Gul/SG/sheeraz/result_archive/MultilabelP50/max_f1_0.59.pth',None)
   

# --------------------------------------------------------------------------------


if __name__ == '__main__':
 runTrain()  



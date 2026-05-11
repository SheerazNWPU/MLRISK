# Adaptive Deep Learning for Slide-Level Multilabel Biomarker Prediction in Breast Cancer WSI Images via Misprediction Risk Analysis

This project contains the code for our work titled **"Adaptive Deep Learning for Slide-Level Multilabel Biomarker Prediction in Breast Cancer WSI Images via Misprediction Risk Analysis."** This code enables the detection of mispredictions in any multilabel task trained using any model. In our work, we focus on multilabel biomarker prediction for breast cancer. Our general experiments use ResNet50 as the baseline model, which can be replaced with any deep neural network (DNN) model, including Transformers and Graph Neural Networks.


## Overall Framework
The overall framework of our work is shown below:

![Risk Model Drawing](Risk%20Model%20Drawing.png)

## Data Usage  

- **Early Breast Cancer Core-Needle Biopsy WSI Dataset (BCNB)**  
  [BCNB Dataset](https://bcnb.grand-challenge.org/)  

- **Post-NAT BRCA Dataset**  
  [Post-NAT BRCA](https://www.cancerimagingarchive.net/collection/post-nat-brca/)  

- **Histopathology Images for End-to-End AI (Based on TCGA-BRCA)**  
  [TCGA-BRCA Dataset](https://zenodo.org/records/5337009)  


## Installation
Install the required packages listed in `Requirements.txt`.

## Usage
```bash
Feature Processing
PrepareRiskData
OneSidedRules
Common
python Main.py

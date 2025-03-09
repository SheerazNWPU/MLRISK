# Adaptive Deep Learning for Slide-Level Multilabel Biomarker Prediction in Breast Cancer WSI Images via Misprediction Risk Analysis

This project represents the code of our work titled **"Adaptive Deep Learning for Slide-Level Multilabel Biomarker Prediction in Breast Cancer WSI Images via Misprediction Risk Analysis"**. This code can detect mispredictions for any multilabel task trained using any model. In our work, we chose multilabel biomarker prediction for breast cancer as the task. Our general experiments use ResNet50 as the baseline model, which can be replaced by any DNN model, including Transformers and Graph Neural Networks.

## Overall Framework
The overall framework of our work is shown below:

![Risk Model Drawing](Risk%20Model%20Drawing.png)

## Comparison Results

The table below shows the comparison results between our method and state of the art methods.

| Model       | ER  | PR  | HER2 | F1 ± SD       | Precision ± SD | Recall ± SD  | Hamming Loss ± SD |
|------------|-----|-----|------|--------------|---------------|-------------|------------------|
| Resnet50   | 0.73 | 0.68 | 0.66 | 0.64 ± 0.24 | 0.60 ± 0.22  | 0.69 ± 0.28 | 0.26 ± 0.03  |
| GCN        | 0.71 | 0.69 | 0.70 | 0.57 ± 0.40 | 0.50 ± 0.35  | 0.66 ± 0.47 | 0.44 ± 0.07  |
| VIT        | 0.52 | 0.50 | 0.66 | 0.47 ± 0.19 | 0.58 ± 0.25  | 0.39 ± 0.16 | 0.38 ± 0.25  |
| TransPath  | 0.64 | 0.62 | 0.61 | 0.57 ± 0.28 | 0.55 ± 0.26  | 0.59 ± 0.29 | 0.38 ± 0.31  |
| TransMIL   | 0.66 | 0.61 | 0.58 | 0.57 ± 0.40 | 0.50 ± 0.35  | 0.67 ± 0.47 | 0.31 ± 0.14  |
| CLAM       | 0.73 | 0.67 | 0.68 | 0.64 ± 0.25 | 0.61 ± 0.21  | 0.67 ± 0.29 | 0.31 ± 0.02  |
| ReceptorNet| 0.77 | 0.71 | 0.24 | **0.70 ± 0.22** | 0.58 ± 0.24  | **1.00 ± 0.00** | 0.42 ± 0.24  |
| DAMLN      | **0.79** | 0.73 | 0.20 | 0.69 ± 0.22 | 0.57 ± 0.23  | **1.00 ± 0.00** | 0.43 ± 0.23  |
| LearnRisk  | 0.73 | 0.68 | 0.67 | 0.62 ± 0.27 | 0.59 ± 0.25  | 0.67 ± 0.31 | 0.31 ± 0.03  |
| MLRisk     | 0.74 | **0.73** | **0.70** | 0.66 ± 0.21 | **0.63 ± 0.22** | 0.70 ± 0.28 | **0.23 ± 0.03** |

**Table:** The comparative evaluation results of adaptive learning on the BCBM Dataset.


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
StartCodebox
PreTraining
PrepareRiskData
OneSidedRules
Common
python Main.py 123
EndCodeBox
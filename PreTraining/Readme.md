# Training and Feature Processing Pipeline

## Steps to Train and Process Features

### 1. Pre-Train the Model
Train the model for a certain number of epochs:
```bash
python PreTraining.py
```

### 2. Extract Features and Results
Obtain the features and results in CSV distributions:
```bash
python GetDistribution.py
```

### 3. Process Features and Build Label Dependency
Process the extracted features to construct label dependencies and fuse them together:
```bash
python FeatureProccessing.py
```


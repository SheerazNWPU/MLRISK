import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchro
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
torch.set_num_threads(4) 
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import cosine_similarity
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import ExpectationMaximization
#import shap
import os
import torch.nn.functional as F
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import KBinsDiscretizer
import gc
from tqdm import tqdm
import traceback
#import lime
#from lime.lime_tabular import LimeTabularExplainer  # Make sure this line is added


# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

# -------- Model Definitions --------
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.attention_bn = nn.Linear(feature_dim, 1)  # Attention for Bayesian network features
        self.attention_me = nn.Linear(feature_dim, 1)  # Attention for Mixture of Experts features
        self.sigmoid = nn.Sigmoid()  # To normalize attention scores

    def forward(self, features_bn, features_me):
        # Compute attention scores
        weight_bn = self.sigmoid(self.attention_bn(features_bn))  # Shape: (batch_size, 1)
        weight_me = self.sigmoid(self.attention_me(features_me))  # Shape: (batch_size, 1)

        # Normalize weights
        total_weight = weight_bn + weight_me
        weight_bn = weight_bn / total_weight
        weight_me = weight_me / total_weight

        # Weighted fusion
        fused_features = weight_bn * features_bn + weight_me * features_me
        return fused_features
        
        
class AttentionDependencyModel(nn.Module):
    def __init__(self, feature_dim, label_dim, embed_dim=512, num_heads=4, num_layers=2):
        super(AttentionDependencyModel, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim + embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

    def forward(self, features, labels):
        label_embeds = self.label_embedding(labels)
        combined_input = torch.cat([features, label_embeds], dim=-1)
        combined_input = combined_input.unsqueeze(1)
        transformed = self.transformer(combined_input)
        return transformed[:, 0, :]
        
class MoEFeatureDependencyModelMultilabel(nn.Module):
    def __init__(self, feature_dim, label_dim, num_experts, embed_dim=512, output_dim=2048):
        super(MoEFeatureDependencyModelMultilabel, self).__init__()
        
        # Label embedding layer (embedding the labels)
        self.label_embedding = nn.Embedding(label_dim, embed_dim)
        
        # Gating layer to assign probabilities to experts
        self.gating = nn.Linear(feature_dim + embed_dim, num_experts)
        
        # Expert layers (each outputs a vector of size output_dim)
        self.experts = nn.ModuleList([
            nn.Linear(feature_dim + embed_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, features, labels):
        # Ensure labels are the same dtype as label embeddings
        labels = labels.to(self.label_embedding.weight.dtype)
    
        # Compute label embeddings
        label_embeds = torch.matmul(labels, self.label_embedding.weight)  # Shape: [batch_size, embed_dim]
    
        # Combine features with label embeddings
        combined_input = torch.cat([features, label_embeds], dim=-1)  # Shape: [batch_size, feature_dim + embed_dim]
    
        # Gating mechanism
        gating_weights = torch.sigmoid(self.gating(combined_input))  # Shape: [batch_size, num_experts]
    
        # Expert outputs
        expert_outputs = torch.stack([expert(combined_input) for expert in self.experts], dim=1)  # Shape: [batch_size, num_experts, output_dim]
    
        # Weighted sum of expert outputs
        transformed_features = torch.einsum('bi,bij->bj', gating_weights, expert_outputs)  # Shape: [batch_size, output_dim]
        return transformed_features




class ContrastiveTransform(nn.Module):
    def __init__(self):
        super(ContrastiveTransform, self).__init__()
        # Adjust the input size to match ResNet50 output dimensions
        self.linear = nn.Linear(2048, 512)  
        self.relu = nn.ReLU()

    def forward(self, features):
        return self.relu(self.linear(features))

# Wrap the MoE model to work for LIME (model should not require gradients for LIME)
class WrapperModelForLIME:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.set_device(device)  # Set the device after initialization

    def predict(self, x):
        # Ensure x is a tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a tensor, but got {type(x)}.")

        # Move input tensor to the same device as the model
        x = x.to(self.device)

        # Perform forward pass with no gradient tracking
        with torch.no_grad():
            outputs = self.model(x)  # Forward pass with features (no labels)

        probabilities = torch.softmax(outputs, dim=-1)
        return probabilities.cpu().numpy()

    def forward(self, x):
        # In the case of LIME, we only want to pass x without labels
        return self.model(x, torch.zeros(x.size(0), 3).to(x.device))  # Empty labels

    def eval(self):
        """Set the wrapped model to evaluation mode."""
        self.model.eval()  # Call eval() on the underlying model

        
def build_feature_transform_network(feature_dim=2048):
    model = BayesianNetwork()
    model.add_nodes_from(['transformed_feature', 'label1', 'label2', 'label3'])
    
    model.add_edge('transformed_feature', 'label1')
    model.add_edge('label1', 'label2')
    model.add_edge('label2', 'label3')
    
    feature_values = np.array([[0.33], [0.34], [0.33]])
    feature_cpd = TabularCPD(
        variable='transformed_feature',
        variable_card=3,
        values=feature_values
    )
    
    label1_values = np.array([
        [0.7, 0.5, 0.3],
        [0.3, 0.5, 0.7]
    ])
    label1_cpd = TabularCPD(
        variable='label1',
        variable_card=2,
        values=label1_values,
        evidence=['transformed_feature'],
        evidence_card=[3]
    )
    
    label2_values = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
    ])
    label2_cpd = TabularCPD(
        variable='label2',
        variable_card=2,
        values=label2_values,
        evidence=['label1'],
        evidence_card=[2]
    )
    
    label3_values = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
    ])
    label3_cpd = TabularCPD(
        variable='label3',
        variable_card=2,
        values=label3_values,
        evidence=['label2'],
        evidence_card=[2]
    )
    
    model.add_cpds(feature_cpd, label1_cpd, label2_cpd, label3_cpd)
    model.check_model()
    
    return model

def transform_features_bayesian(bayesian_model, features_df, labels_df):
    inference = VariableElimination(bayesian_model)
    transformed_features = []
    
    batch_size = 100
    n_features = features_df.shape[1]
    
    for i in tqdm(range(0, len(features_df), batch_size), desc="Bayesian transformation"):
        batch_features = features_df.iloc[i:i+batch_size]
        batch_transformed = []
        
        for _, features in batch_features.iterrows():
            row_transformed = []
            for feat_idx in range(n_features):
                feature_val = features[feat_idx]
                
                if feature_val <= np.percentile(features_df.iloc[:, feat_idx], 33):
                    evidence = {'transformed_feature': 0}
                elif feature_val <= np.percentile(features_df.iloc[:, feat_idx], 66):
                    evidence = {'transformed_feature': 1}
                else:
                    evidence = {'transformed_feature': 2}
                
                try:
                    label_probs = []
                    for label in ['label1', 'label2', 'label3']:
                        result = inference.query([label], evidence=evidence)
                        label_probs.append(result.values[1])
                    
                    weights = np.array([0.5, 0.3, 0.2])
                    transformed_val = np.dot(label_probs, weights) * feature_val
                    row_transformed.append(transformed_val)
                    
                except Exception as e:
                    row_transformed.append(feature_val)
            
            batch_transformed.append(row_transformed)
            
        transformed_features.extend(batch_transformed)
        
        # Clear memory periodically
        if i % 1000 == 0:
            gc.collect()
    
    return pd.DataFrame(transformed_features)


def compute_feature_importance(model, features, labels, device, batch_size=32):
    model.eval()
    feature_importance_scores = []
    num_batches = (features.shape[0] + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Computing feature importance"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, features.shape[0])
        
        batch_features = features[start_idx:end_idx].detach().clone()
        batch_labels = labels[start_idx:end_idx]
        
        batch_features.requires_grad_(True)
        model.zero_grad()
        
        output = model(batch_features, batch_labels)
        output.sum().backward()
        
        # Store the gradients for each instance in the batch
        batch_importance = batch_features.grad.abs().cpu().numpy()
        feature_importance_scores.append(batch_importance)
        
        # Clear memory
        del batch_features, batch_labels, output
        torch.cuda.empty_cache()

    # Concatenate all batches
    feature_importance_scores = np.concatenate(feature_importance_scores, axis=0)
    return feature_importance_scores
import torch
import pandas as pd
import numpy as np
import os
import gc
import traceback

def process_and_save_features(main_path, features_csv, labels_csv, output_prefix, batch_size=32):
    print(f"\nProcessing dataset: {output_prefix}")
    
    # Build paths
    features_path = os.path.join(main_path, features_csv)
    labels_path = os.path.join(main_path, labels_csv)
    output_path = main_path

    # Load data in chunks
    print("Loading data...")
    chunk_size = 10000
    features_chunks = pd.read_csv(features_path, header=None, chunksize=chunk_size)
    features_df = pd.concat([chunk for chunk in features_chunks])
    labels_df = pd.read_csv(labels_path, header=None)
    
    # Convert to tensors
    features = torch.tensor(features_df.values, dtype=torch.float32)
    labels = torch.tensor(labels_df.values, dtype=torch.long)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Move data to device
        features = features.to(device)
        labels = labels.to(device)

        # 1. Attention-Based Transformation
        print("Performing attention-based transformation...")
        attention_model = AttentionDependencyModel(feature_dim=features.shape[1], label_dim=3).to(device)
        with torch.no_grad():
            attention_features = attention_model(features, labels.argmax(dim=1))

        # 2. Contrastive Transformation
        print("Performing contrastive transformation...")
        contrastive_transform = ContrastiveTransform().to(device)
        with torch.no_grad():
            contrastive_features = contrastive_transform(features)

        
        # 4. Feature Importance Using MoE
        print("Computing MoE feature importance...")
        moe_model = MoEFeatureDependencyModelMultilabel(
            feature_dim=features.shape[1],
            label_dim=3,
            num_experts=5
        ).to(device)
        
        feature_importance_scores = compute_feature_importance(
            moe_model, features, labels, device, batch_size
        )
        
        feature_importance_df = pd.DataFrame(
            feature_importance_scores,
            columns=None  # Explicitly specify no columns
        )
        #print(feature_importance_df)
        # Save transformed features
        
        # 3. Bayesian Network Transformation
        print("Performing Bayesian network transformation...")
        bayesian_model = build_feature_transform_network()
        bayesian_features = transform_features_bayesian(bayesian_model, features_df, labels_df)
        print(bayesian_features.shape)
        print("Performing Feature Fusion...")
        feature_dim = 64
        attention_fusion = AttentionFusion(feature_dim)
        bayesian_features = torch.tensor(bayesian_features.values, dtype=torch.float32)
        moe_feature_tensor = torch.tensor(feature_importance_scores, dtype=torch.float32)
        fused_features = attention_fusion(bayesian_features, moe_feature_tensor)
        fused_features_df = pd.DataFrame(fused_features.detach().cpu().numpy())
        bayesian_features_df = pd.DataFrame(bayesian_features.detach().cpu().numpy())
        # Save transformed features
        print("Saving transformed features...")
        os.makedirs(output_path, exist_ok=True)
        
        transformed_data = {
            "att": pd.DataFrame(attention_features.cpu().numpy()),
            "con": pd.DataFrame(contrastive_features.cpu().numpy()),
            "bay": bayesian_features_df,
            "moe": feature_importance_df,
            "b-m" : fused_features_df
        }
        
        for prefix, df in transformed_data.items():
            file_path = os.path.join(output_path, f"distribution_{prefix}_{output_prefix}.csv")
            # Save in chunks with no headers
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                mode = 'w' if i == 0 else 'a'
                chunk.to_csv(file_path, mode=mode, header=False, index=False)

        print(f"Successfully saved all transformed features for {output_prefix}")

    except Exception as e:
        print(f"Error processing {output_prefix}: {e}")
        traceback.print_exc()

    finally:
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    main_path = "/YourSavePath/result_archive/Multilabel50/"

    datasets = {
        "train": {"features": "distribution_x4_train.csv", "labels": "targets_train.csv", "output": "train"},
        "test": {"features": "distribution_x4_test.csv", "labels": "targets_test.csv", "output": "test"},
        "val": {"features": "distribution_x4_val.csv", "labels": "targets_val.csv", "output": "val"}
    }

    for dataset, paths in datasets.items():
        try:
            process_and_save_features(main_path, paths["features"], paths["labels"], paths["output"])
            print(f"Successfully processed {dataset} dataset")
        except Exception as e:
            print(f"Error processing {dataset} dataset: {e}")
            continue
from __future__ import print_function, with_statement, division, absolute_import

import random

import torch
import torchsnooper
from torch import nn, optim
from torch.distributions.normal import Normal
from common import config
import numpy as np
from scipy import sparse as sp
from collections import Counter
import math
import logging
from tqdm import trange, tqdm
import numpy as np
import os
from typing import List, Optional, Union
from sklearn.metrics import roc_curve, auc
from risker.pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import LambdaLR
import torch
import scipy.stats as stats
from scipy import stats
from scipy.stats import spearmanr
from torch.distributions import MultivariateNormal
#from torch.distributions.multivariate_normal import MultivariateNormal
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = [0,1]  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection)
class_num = cfg.get_class_num()

LEARN_VARIANCE = cfg.learn_variance
APPLY_WEIGHT_FUNC = cfg.apply_function_to_weight_classifier_output

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(module)s:%(levelname)s] - %(message)s")


class LabelEmbedding(nn.Module):
    def __init__(self, num_labels, embed_dim):
        super(LabelEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_labels, embed_dim)
        
    def forward(self, labels):
        return self.embedding(labels)

def convert_tensor(original_tensor):
    # Convert to probabilities using sigmoid
    probabilities = torch.sigmoid(original_tensor)
    
    # Create the complement probabilities
    complement_probs = 1 - probabilities
    
    # Stack the probabilities and their complements
    result_tensor = torch.stack((probabilities, complement_probs), dim=-1)
    
    return result_tensor

# Sheeraz 16/8/2024
def calculate_label_weights(labels, num_labels):
    """
    Calculate label weights for a multilabel classification problem.
    
    Parameters:
    labels (np.ndarray): A 2D array of shape (num_samples, num_labels) with binary labels.
    num_labels (int): The number of labels.
    
    Returns:
    torch.Tensor: A tensor of shape (num_labels,) with the calculated weights for each label.
    """
    # Ensure labels is a 2D array
    labels = np.array(labels).astype(int)
    # Calculate the total number of samples
    total_samples = labels.shape[0]
    # Calculate positive counts for each label
    positive_counts = np.sum(labels, axis=0)
    # Calculate negative counts for each label
    negative_counts = total_samples - positive_counts
    # Avoid division by zero by setting minimum count to 1
    positive_counts[positive_counts == 0] = 1
    negative_counts[negative_counts == 0] = 1
    # Calculate weights: inverse of label frequency for both positive and negative classes
    positive_weights = 1.0 / positive_counts
    negative_weights = 1.0 / negative_counts
    # Combine weights: average of positive and negative weights
    label_weights = (positive_weights + negative_weights) / 2
    # Normalize weights
    label_weights = label_weights / label_weights.sum()
    #print(label_weights)
    return torch.tensor(label_weights, dtype=torch.float32)


## Updated Sheeraz 7/6/2024

def my_truncated_normal_ppf(confidence, a, b, mean, stddev, class_num):
    # Reshape mean and stddev
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
    
    norm = Normal(mean, stddev)
    
    # Compute values for CDF
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)

    # Compute values for icdf input
    icdf_input_1 = confidence * _sb + _sa * (1.0 - confidence)
    icdf_input_2 = confidence * _nb + _na * (1.0 - confidence)

    # Clip the values to avoid numerical instability
    icdf_input_1 = torch.clamp(icdf_input_1, min=1e-6, max=1-1e-6)
    icdf_input_2 = torch.clamp(icdf_input_2, min=1e-6, max=1-1e-6)

    # Compute the final output using icdf
    y = torch.where(a > 0,
                    -norm.icdf(icdf_input_1),
                    norm.icdf(icdf_input_2))

    return torch.reshape(y, (-1, class_num))


def ensure_positive_definite(cov_matrix, epsilon=1e-6):
    """
    Ensure that a matrix is positive definite by trying Cholesky decomposition.
    If the matrix is not positive definite, adjust it by increasing the diagonal elements.
    
    Args:
        cov_matrix (torch.Tensor): The covariance matrix.
        epsilon (float): Small value added to the diagonal if the matrix is not positive definite.
    
    Returns:
        torch.Tensor: A positive definite covariance matrix.
    """
    try:
        # Try Cholesky decomposition to check if the matrix is positive definite
        torch.linalg.cholesky(cov_matrix)
    except RuntimeError:
        # If decomposition fails, the matrix is not positive definite
        # Increase diagonal elements by epsilon until it works
        diag_addition = torch.eye(cov_matrix.size(0), device=cov_matrix.device) * epsilon
        cov_matrix += diag_addition
        print(f"Covariance matrix adjusted by adding {epsilon} to diagonal elements for stability.")
    
    return cov_matrix
def calculate_evar(simulations, alpha, max_lambda=10.0, num_steps=1000):
    """
    Calculate the Entropic Value at Risk (EVaR) for a given set of simulations.
    
    :param simulations: A tensor of simulated outcomes.
    :param alpha: Confidence level for EVaR (e.g., 0.95).
    :param max_lambda: Maximum value of lambda for the optimization process.
    :param num_steps: Number of steps for lambda optimization.
    :return: EVaR value.
    """
    # Initialize lambda and the step size
    lambdas = torch.linspace(1e-5, max_lambda, num_steps)
    evars = []

    # Calculate EVaR for each lambda
    for lam in lambdas:
        exp_moment = torch.exp(lam * simulations).mean(dim=0)  # Exponential moment of the losses
        evar_lam = (1 / lam) * (torch.log(exp_moment) - torch.log(torch.tensor(alpha)))
        evars.append(evar_lam)
    
    # Return the minimum EVaR over all lambda values
    evars = torch.stack(evars, dim=0)
    evar = torch.min(evars, dim=0).values
    return evar   



def pairwise_correlation(machine_mus: torch.Tensor) -> torch.Tensor:
    """Calculate the pairwise correlation matrix across labels (or features if needed).
    
    Args:
        machine_mus: Input tensor [num_samples, num_features, num_labels] or [num_samples, num_labels]
    
    Returns:
        torch.Tensor: Pairwise correlation matrix across labels
    """
    # If input is 2D, add an extra dimension for features (i.e., [num_samples, 1, num_labels])
    if machine_mus.dim() == 2:
        machine_mus = machine_mus.unsqueeze(1)  # Add a dimension for features

    # Ensure input is 3D after adjustment
    if machine_mus.dim() != 3:
        raise ValueError(f"Expected input tensor to have 3 dimensions, got {machine_mus.dim()}")

    # Compute correlation across labels
    num_labels = machine_mus.shape[-1]
    
    # Calculate pairwise correlations
    machine_mus_flat = machine_mus.reshape(-1, num_labels)  # Flatten features and samples
    corr_matrix = torch.corrcoef(machine_mus_flat.T)
    
    return corr_matrix



def shrinkage_correlation(machine_mus: torch.Tensor, shrinkage_factor: float = 0.2) -> torch.Tensor:
    """Calculate correlation matrix with shrinkage over labels (not features).
    
    Args:
        machine_mus: Input tensor [num_samples, num_features, num_labels]
        shrinkage_factor: Shrinkage intensity (0-1)
        
    Returns:
        torch.Tensor: Shrunk correlation matrix across labels
    """
    # Ensure input is 3D: [num_samples, num_features, num_labels]
    if machine_mus.dim() != 3:
        raise ValueError(f"Expected input tensor to have 3 dimensions, got {machine_mus.dim()}")
    
    # Compute the pairwise correlation across labels
    pairwise_corr = pairwise_correlation(machine_mus)
    
    # Apply shrinkage to the correlation matrix
    target = torch.eye(pairwise_corr.shape[0], device=pairwise_corr.device)
    return (1 - shrinkage_factor) * pairwise_corr + shrinkage_factor * target


def adaptive_correlation(machine_mus: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Calculate correlation matrix with thresholding across labels (not features).
    
    Args:
        machine_mus: Input tensor [num_samples, num_features, num_labels]
        threshold: Correlation threshold to apply
        
    Returns:
        torch.Tensor: Thresholded correlation matrix across labels
    """
    # Ensure input is 3D: [num_samples, num_features, num_labels]
    if machine_mus.dim() != 3:
        raise ValueError(f"Expected input tensor to have 3 dimensions, got {machine_mus.dim()}")
    
    # Compute pairwise correlation across labels
    pairwise_corr = pairwise_correlation(machine_mus)
    
    # Apply threshold to the correlation matrix
    mask = torch.abs(pairwise_corr) < threshold
    pairwise_corr[mask] = 0
    return pairwise_corr



def adaptive_correlation(machine_mus: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Calculate correlation matrix with thresholding
    
    Args:
        machine_mus: Input tensor
        threshold: Correlation threshold
        
    Returns:
        torch.Tensor: Thresholded correlation matrix
    """
    pairwise_corr = pairwise_correlation(machine_mus)
    mask = torch.abs(pairwise_corr) < threshold
    pairwise_corr[mask] = 0
    return pairwise_corr

def check_normality(data, min_samples=8):
    """
    Check normality of data using D'Agostino and Pearson's test
    """
    num_labels = data.shape[-1]
    is_normal = []
    p_values = []
    
    for i in range(num_labels):
        label_data = data[..., i].numpy()
        
        # Check if we have enough samples
        if label_data.shape[0] < min_samples:
            # If not enough samples, assume normal distribution
            print(f"Warning: Label {i} has {label_data.shape[0]} samples (less than {min_samples}). Assuming normal distribution.")
            is_normal.append(True)
            p_values.append(1.0)
        else:
            try:
                _, p_value = stats.normaltest(label_data, axis=0)
                is_normal.append(all(p > 0.05 for p in p_value))
                p_values.append(p_value)
            except Exception as e:
                print(f"Warning: Normality test failed for label {i}: {str(e)}. Assuming normal distribution.")
                is_normal.append(True)
                p_values.append(1.0)
    
    return is_normal, p_values

def get_best_ppf_method(data):
    """
    Determine the best PPF method based on data characteristics
    """
    num_labels = data.shape[-1]
    methods = []
    
    try:
        is_normal_per_label, p_values_per_label = check_normality(data)
    except Exception as e:
        print(f"Warning: Normality test failed: {str(e)}. Using default methods.")
        # Default to a mix of methods
        return ['univariate'] * num_labels
    
    for label_idx in range(num_labels):
        if is_normal_per_label[label_idx]:
            # If data appears normal, use univariate method
            methods.append('univariate')
        else:
            # If non-normal, use empirical or mixture method
            # Could add more sophisticated selection logic here
            methods.append('empirical')
    
    return methods

def calculate_covariance_matrix(stddev_matrix: torch.Tensor, corr_matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates the covariance matrix from standard deviations and correlations.
    Handles various input dimension cases safely.

    Args:
        stddev_matrix (torch.Tensor): Tensor of standard deviations [batch_size, num_features] or [batch_size, num_features, 1]
        corr_matrix (torch.Tensor): Tensor of correlations [batch_size, num_features, num_features] or [num_windows, num_features, num_features]

    Returns:
        torch.Tensor: Covariance matrix [batch_size, num_features, num_features]
    """
    # First ensure stddev_matrix is at least 2D
    if stddev_matrix.dim() == 1:
        stddev_matrix = stddev_matrix.unsqueeze(0)
    
    # Remove any trailing singleton dimensions from stddev_matrix
    while stddev_matrix.dim() > 2 and stddev_matrix.size(-1) == 1:
        stddev_matrix = stddev_matrix.squeeze(-1)
    
    # Ensure corr_matrix is 3D
    if corr_matrix.dim() == 2:
        corr_matrix = corr_matrix.unsqueeze(0)
    
    # Get the correct dimensions
    batch_size = stddev_matrix.size(0)
    stddev_features = stddev_matrix.size(1)
    corr_features = corr_matrix.size(-1)
    
    # Verify feature dimensions match
    if stddev_features != corr_features:
        raise ValueError(f"Feature dimension mismatch: stddev has {stddev_features} features, "
                        f"correlation has {corr_features} features")
    
    # Handle case where correlation matrix has more entries (from windows)
    if corr_matrix.size(0) != batch_size:
        # Calculate windows per batch
        windows_per_batch = corr_matrix.size(0) // batch_size
        if windows_per_batch > 0:
            # Reshape and average windows
            corr_matrix = corr_matrix.view(batch_size, windows_per_batch, corr_features, corr_features)
            corr_matrix = corr_matrix.mean(dim=1)
        else:
            # If we have fewer windows than batch size, repeat the correlation matrix
            corr_matrix = corr_matrix.expand(batch_size, -1, -1)
    
    # Reshape stddev_matrix for matrix multiplication
    stddev_matrix = stddev_matrix.unsqueeze(-1)  # [batch_size, num_features, 1]
    
    # Compute covariance matrix
    stddev_outer = stddev_matrix @ stddev_matrix.transpose(-2, -1)  # [batch_size, num_features, num_features]
    cov_matrix = stddev_outer * corr_matrix
    
    return cov_matrix


def gaussian_function(a, b, c, x):
    _part = (- torch.div((x - b) ** 2, 2.0 * (c ** 2)))
    _f = -torch.exp(_part) + a + 1.0
    return _f


def L1loss(model, beta=0.001):
    l1_loss = torch.tensor(0., requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(parma)).to(device[0])
    return l1_loss.to(device[0])


def L2loss(model, alpha=0.001):
    l2_loss = torch.tensor(0., requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(parma ** 2)).to(device[0])
    return l2_loss.to(device[0])


class RiskModel(nn.Module):
    def __init__(self, rule_m, max_w, embed_dim, label_num, label_true, num_heads=5, init_variance=None):
        super(RiskModel, self).__init__()
        self.max_w = max_w
        alaph = torch.tensor(cfg.risk_confidence, dtype=torch.float32)
        #print(alaph)
        a = torch.tensor(0., dtype=torch.float32)
        b = torch.tensor(1., dtype=torch.float32)

        self.label_num = label_num
        
        weight_fun_b = torch.tensor([0.5], dtype=torch.float32)

        self.register_buffer('alaph', alaph)
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('weight_fun_b', weight_fun_b)

        self.rule_m = rule_m
        self.machine_m = cfg.interval_number_4_continuous_value

        self.rule_var = nn.Parameter(
            torch.empty((1, self.rule_m,), dtype=torch.float32, requires_grad=True)
        )
        torch.nn.init.uniform_(self.rule_var, 0, 1)
        self.machine_var = nn.Parameter(
            torch.empty(1, self.machine_m, dtype=torch.float32, requires_grad=True)
        )
        torch.nn.init.uniform_(self.machine_var, 0, 1)

        self.rule_w = nn.Parameter(
            torch.empty((1, self.rule_m,), dtype=torch.float32, requires_grad=True)
        )
        torch.nn.init.uniform_(self.rule_w, 0, max_w)

        self.learn2rank_sigma = nn.Parameter(torch.tensor(1., dtype=torch.float32, requires_grad=True))
        self.risk_weight_learn = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))
        self.weight_fun_a = nn.Parameter(torch.tensor([1.], dtype=torch.float32, requires_grad=True))
        self.weight_fun_c = nn.Parameter(torch.tensor([0.5], dtype=torch.float32, requires_grad=True))

        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim*2, num_heads=num_heads, batch_first=True)
        self.attention_to_class = nn.Linear(embed_dim*2, label_num)
        
        # Label embeddings
        self.label_embedding = LabelEmbedding(label_num, embed_dim)
        self.label_weights = calculate_label_weights(label_true, label_num)

    def forward(self, machine_labels, rule_mus, machine_mus, rule_feature_matrix, machine_feature_matrix,
                machine_one, y_risk, y_mul_risk, init_rule_mu, init_machine_mu, use_label_weights=True):
        
        # Prepare label embeddings
        label_indices = torch.arange(self.label_num).to(machine_labels.device)
        label_embeds = self.label_embedding(label_indices)
        
        # Attention mechanism on machine_mus
        if machine_mus.dim() == 2:
            machine_mus = machine_mus.unsqueeze(1)
        machine_mus = torch.cat((machine_mus, label_embeds.unsqueeze(0).expand(machine_mus.size(0), -1, -1)), dim=-1)
        attn_output, _ = self.multihead_attn(machine_mus, machine_mus, machine_mus)
        class_projection = self.attention_to_class(attn_output.squeeze(1))
        machine_w = torch.sigmoid(class_projection)

        machine_w = machine_w[:, :, 0]

        machine_mus_vector = torch.reshape(torch.sum(machine_mus, 2), (-1, self.label_num))

        big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * machine_w + 1e-10

        new_rule_mus = torch.from_numpy(init_rule_mu).clone().to(machine_labels.device).float()
        new_mac_mus = torch.from_numpy(init_machine_mu).clone().to(machine_labels.device).float()
        new_rule_mus[torch.where(new_rule_mus < 0.1)] = 0.1
        new_mac_mus[torch.where(new_mac_mus < 0.1)] = 0.1
        rule_standard_deviation = new_rule_mus * self.rule_var
        mac_standard_deviation = new_mac_mus * self.machine_var
        rule_var = rule_standard_deviation ** 2
        machine_var = mac_standard_deviation ** 2

        rule_sigma = rule_feature_matrix * rule_var
        machine_sigma = machine_feature_matrix * machine_var
        machine_sigma_vector = torch.sum(machine_sigma, 2).reshape((-1, self.label_num))

        big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (machine_w ** 2) + 1e-10
        r_sig = torch.sum(rule_sigma * (self.rule_w ** 2), 2)
        m_sig = machine_sigma_vector * (machine_w ** 2)

        weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) + machine_w + 1e-10
        big_mu = big_mu / (weight_vector + 1e-10)
        big_sigma = big_sigma / (weight_vector ** 2 + 1e-10)

        Fr_alpha = my_truncated_normal_ppf(self.alaph, self.a, self.b, big_mu, torch.sqrt(big_sigma), self.label_num)
        Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alaph, self.a, self.b, big_mu, torch.sqrt(big_sigma), self.label_num)
        prob_mul = 1 - Fr_alpha_bar

        prob_mul1 = convert_tensor(prob_mul)
        
        prob = prob_mul1 * machine_one
        prob = prob.sum(dim=-1)

        return prob_mul, prob, [self.weight_fun_a.data, self.weight_fun_b.data, self.weight_fun_c.data], \
               self.rule_w.data, rule_var.data, machine_var.data, big_mu, big_sigma, \
               torch.sum(rule_feature_matrix * self.rule_w, 2), machine_mus_vector, \
               [Fr_alpha, Fr_alpha_bar], r_sig, m_sig
    def get_label_weights(self):
        return self.label_weights
'''
class PairwiseLoss(nn.Module):
    def __init__(self, learn2rank_sigma, risk_weight_learn):
        super(PairwiseLoss, self).__init__()
        self.learn2rank_sigma = learn2rank_sigma
        self.result = torch.empty((0, 2), dtype=torch.float32)
        self.init_result = self.result

    # @torchsnooper.snoop()
    def forward(self, input, target):
        print('input')
        print(input)
        print('target')
        print(target)
        pairwise_probs = self.get_pairwise_combinations(input).to(device[0])
        # print(pairwise_probs.shape)
        pairwise_labels = self.get_pairwise_combinations(target.float()).to(device[0])
        # print(pairwise_labels.shape)

        p_target_ij = 0.5 * (1.0 + pairwise_labels[:, 0] - pairwise_labels[:, 1])
        o_ij = pairwise_probs[:, 0] - pairwise_probs[:, 1]

        diff_label_indices = torch.nonzero(p_target_ij != 0.5)  # .squeeze()
        # print(diff_label_indices.shape)
        new_p_target_ij = p_target_ij[diff_label_indices]
        # print(new_p_target_ij.shape)
        
        new_o_ij = o_ij[diff_label_indices] * self.learn2rank_sigma
        # print(self.learn2rank_sigma)
        return torch.sum(- new_p_target_ij * new_o_ij + torch.log(1.0 + torch.exp(new_o_ij))).to(device[0])

    # @torchsnooper.snoop()
    def get_pairwise_combinations(self, input):
        self.result = self.init_result
        for i in range(input.shape[0] - 1):
            tensor = torch.stack(
                torch.meshgrid(input[i, 0], input[i + 1:, 0]), dim=-1
            ).reshape((-1, 2))
            self.result = torch.cat((self.result.to(device[0]), tensor.to(device[0])), dim=0)

        return self.result
'''
'''
class PairwiseLoss(nn.Module):
    def __init__(self, learn2rank_sigma, risk_weight_learn):
        super(PairwiseLoss, self).__init__()
        self.learn2rank_sigma = learn2rank_sigma
        self.risk_weight_learn = risk_weight_learn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        #print('input')
        #print(input)
        #print('target')
        #print(target)
        pairwise_probs = self.get_pairwise_combinations(input)
        pairwise_labels = self.get_pairwise_combinations(target)

        loss = 0
        num_labels = target.shape[1]
        for label_idx in range(num_labels):
            p_target_ij = 0.5 * (1.0 + pairwise_labels[:, 0, label_idx] - pairwise_labels[:, 1, label_idx])
            o_ij = pairwise_probs[:, 0, label_idx] - pairwise_probs[:, 1, label_idx]

            diff_label_indices = torch.nonzero(p_target_ij != 0.5, as_tuple=True)
            new_p_target_ij = p_target_ij[diff_label_indices]
            new_o_ij = o_ij[diff_label_indices] * self.learn2rank_sigma

            loss += torch.sum(- new_p_target_ij * new_o_ij + torch.log(1.0 + torch.exp(new_o_ij)))

        return loss / num_labels

    def get_pairwise_combinations(self, input):
        combinations = []
        num_instances = input.shape[0]
        for i in range(num_instances - 1):
            for j in range(i + 1, num_instances):
                combinations.append(torch.stack([input[i], input[j]], dim=0))
        return torch.stack(combinations).to(self.device)
'''
class PairwiseLoss(nn.Module):
    def __init__(self, learn2rank_sigma, risk_weight_learn, label_weights):
        super(PairwiseLoss, self).__init__()
        self.learn2rank_sigma = learn2rank_sigma
        self.risk_weight_learn = risk_weight_learn
        self.label_weights = label_weights  # Tensor of shape [num_classes]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        pairwise_probs = self.get_pairwise_combinations(input)
        pairwise_labels = self.get_pairwise_combinations(target)

        loss = 0
        num_labels = target.shape[1]
        for label_idx in range(num_labels):
            p_target_ij = 0.5 * (1.0 + pairwise_labels[:, 0, label_idx] - pairwise_labels[:, 1, label_idx])
            o_ij = pairwise_probs[:, 0, label_idx] - pairwise_probs[:, 1, label_idx]

            # Safeguard for numerical stability
            p_target_ij = torch.clamp(p_target_ij, 1e-10, 1 - 1e-10)
            o_ij = torch.clamp(o_ij, -50, 50)  # Prevent overflow/underflow in exponentiation

            diff_label_indices = torch.nonzero(p_target_ij != 0.5, as_tuple=True)
            new_p_target_ij = p_target_ij[diff_label_indices]
            new_o_ij = o_ij[diff_label_indices] * self.learn2rank_sigma

            # Apply label weights
            weight = self.label_weights[label_idx].to(self.device)
            weighted_loss = - new_p_target_ij * new_o_ij * weight + torch.log(1.0 + torch.exp(new_o_ij)) * weight

            # Check for NaNs in the weighted loss
            if torch.isnan(weighted_loss).any():
                print(f"NaN detected in weighted_loss at label index {label_idx}")
                print(f"new_p_target_ij: {new_p_target_ij}")
                print(f"new_o_ij: {new_o_ij}")

            loss += torch.sum(weighted_loss)

        return loss / num_labels

    def get_pairwise_combinations(self, input):
        combinations = []
        num_instances = input.shape[0]
        for i in range(num_instances - 1):
            for j in range(i + 1, num_instances):
                combinations.append(torch.stack([input[i], input[j]], dim=0))
        return torch.stack(combinations).to(self.device)

def train(model, val, test, init_rule_mu, init_machine_mu, epoch_cnn=0, epoches=200, suffle_data=True):
    data_len = len(val.risk_labels)
    
    # Convert data to tensors
    machine_labels = torch.tensor(val.machine_labels, dtype=torch.int)
    rule_mus = torch.tensor(val.get_risk_mean_X_discrete(), dtype=torch.float32)
    machine_mus = torch.tensor(val.get_risk_mean_X_continue(), dtype=torch.float32)
    rule_feature_activate = torch.tensor(val.get_rule_activation_matrix(), dtype=torch.float32)
    machine_feature_activate = torch.tensor(val.get_prob_activation_matrix(), dtype=torch.float32)
    machine_one = torch.tensor(val.machine_label_2_one, dtype=torch.float32)
    y_risk = torch.tensor(val.risk_labels, dtype=torch.float32)
    y_mul_risk = torch.tensor(val.risk_mul_labels, dtype=torch.float32)
    y_true_one = torch.tensor(val.true_label_2_one, dtype=torch.float32)
    
    # Add label embeddings
    label_indices = torch.tensor(val.machine_labels, dtype=torch.long)
    
    # Setup optimizer, criterion, scheduler
    risk_weight = 0
    l2_reg = 0.001
    bs = 4
    batch_num = data_len // bs + (1 if data_len % bs else 0)
    learning_rate = 0.0005
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = PairwiseLoss(model.learn2rank_sigma, model.risk_weight_learn, model.get_label_weights()).to(device[0])
    #criterion = PairwiseLoss(model.learn2rank_sigma, model.risk_weight_learn).to(device[0])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    early_stopping = EarlyStopping(patience=100, verbose=True)

    for epoch in tqdm(range(epoches), desc="Training"):
        model.train()
        if suffle_data:
            index = np.random.permutation(np.arange(data_len))
        else:
            index = np.arange(data_len)
        
        loss_total = 0.
        n_total = 0.
        outputs_all = torch.empty((0, class_num), dtype=torch.float32, requires_grad=False)
        outputs_mul_all = torch.empty((0, class_num), dtype=torch.float32, requires_grad=False)

        for i in range(batch_num):
            machine_labels_batch = machine_labels[index][bs * i: bs * i + bs].to(device[0])
            rule_mus_batch = rule_mus[index][bs * i: bs * i + bs].to(device[0])
            machine_mus_batch = machine_mus[index][bs * i: bs * i + bs].to(device[0])
            rule_feature_activate_batch = rule_feature_activate[index][bs * i: bs * i + bs].to(device[0])
            machine_feature_activate_batch = machine_feature_activate[index][bs * i: bs * i + bs].to(device[0])
            machine_one_batch = machine_one[index][bs * i: bs * i + bs].to(device[0])
            y_risk_batch = y_risk[index][bs * i: bs * i + bs].to(device[0])
            y_mul_risk_batch = y_mul_risk[index][bs * i: bs * i + bs].to(device[0])
            y_true_one_batch = y_true_one[index][bs * i: bs * i + bs].to(device[0])
            
            # Get label embeddings for the batch
            label_embeddings_batch = model.label_embedding(label_indices[index][bs * i: bs * i + bs].to(device[0]))
            
            outputs_mul, outputs, func_params, rule_w, rule_var, machine_var, big_mu, big_sigma, r_w, m_w, fr, _, _ = model(
                y_true_one_batch,
                rule_mus_batch,
                machine_mus_batch,
                rule_feature_activate_batch,
                machine_feature_activate_batch,
                machine_one_batch,
                y_risk_batch,
                y_mul_risk_batch,
                init_rule_mu,
                init_machine_mu,
                label_embeddings_batch  # Pass label embeddings
            )
            
            l1_loss = torch.tensor(0.).to(device[0])
            l2_loss = torch.tensor(0.).to(device[0])
            beta = torch.tensor(0.001).to(device[0])
            
            # L1 and L2 regularization
            for name, param in model.named_parameters():
                if 'var' not in name:
                    l1_loss += beta * torch.sum(torch.abs(param))
                    l2_loss += (0.5 * beta * torch.sum(param ** 2))
            
            optimizer.zero_grad()
            rank_loss1 = criterion(outputs, y_risk_batch)
            loss = rank_loss1 + l1_loss + l2_loss
            loss.backward()
            optimizer.step()
            
            model.rule_w.data.clamp_(0., )
            model.rule_var.data.clamp_(0., 1.)
            model.machine_var.data.clamp_(0., 1.)
            model.weight_fun_a.data.clamp_(1e-10, )
            model.weight_fun_c.data.clamp_(1e-10, )
            
            loss_total += loss.item()
            n_total += len(outputs_mul)
            outputs_all = torch.cat((outputs_all.to(device[0]), outputs), dim=0)
            outputs_mul_all = torch.cat((outputs_mul_all.to(device[0]), outputs_mul), dim=0)
        
        scheduler.step()
        logging.info("Epoch %d, Loss: %f" % (epoch, loss_total))
        if epoch % 1 == 0:
            print('-----------------test predict')
            predict(model, test, epoch, init_rule_mu, init_machine_mu, epoch_cnn, True)

        if early_stopping.early_stop:
            logging.info('Early stopping')
            break

    return func_params, rule_w, rule_var, machine_var



def predict(model, test, epoch, init_rule_mu, init_machine_mu, epoch_cnn=0, is_print=False):
    data_len = len(test.risk_labels)

    machine_labels = torch.tensor(test.machine_labels, dtype=torch.int)
    
    # Convert test data to tensors
    rule_mus = torch.tensor(test.get_risk_mean_X_discrete(), dtype=torch.float32)
    machine_mus = torch.tensor(test.get_risk_mean_X_continue(), dtype=torch.float32)
    rule_feature_activate = torch.tensor(test.get_rule_activation_matrix(), dtype=torch.float32)
    machine_feature_activate = torch.tensor(test.get_prob_activation_matrix(), dtype=torch.float32)
    machine_one = torch.tensor(test.machine_label_2_one, dtype=torch.int)
    y_risk = torch.tensor(test.risk_labels, dtype=torch.int)
    y_mul_risk = torch.tensor(test.risk_mul_labels, dtype=torch.int)
    
    # Get label embeddings
    test.machine_labels.cuda()
    label_indices = torch.tensor(test.machine_labels, dtype=torch.long)
    label_indices = label_indices.cuda()
    label_embeddings = model.label_embedding(label_indices)  # Ensure `label_embeddings` is a method in your model

    bs = 4
    batch_num = data_len // bs + (1 if data_len % bs else 0)
    outputs_all = torch.empty((0, class_num), dtype=torch.float32)
    outputs_mul_all = torch.empty((0, class_num), dtype=torch.float32)
    
    model.eval()
    right = 0
    right_fr = 0
    right_fr_bar = 0
    right_entropy = 0
    tot = 0
    rs_right = 0
    ms_right = 0
    
    with torch.no_grad():
        for i in range(batch_num):
            machine_labels_batch = machine_labels[bs * i: bs * i + bs].to(device[0])
            rule_mus_batch = rule_mus[bs * i: bs * i + bs].to(device[0])
            machine_mus_batch = machine_mus[bs * i: bs * i + bs].to(device[0])
            rule_feature_activate_batch = rule_feature_activate[bs * i: bs * i + bs].to(device[0])
            machine_feature_activate_batch = machine_feature_activate[bs * i: bs * i + bs].to(device[0])
            machine_one_batch = machine_one[bs * i: bs * i + bs].to(device[0])
            y_risk_batch = y_risk[bs * i: bs * i + bs].to(device[0])
            y_mul_risk_batch = y_mul_risk[bs * i: bs * i + bs].to(device[0])
            label_embeddings_batch = label_embeddings[bs * i: bs * i + bs].to(device[0])
            
            outputs_mul, outputs, _, r_every_w, r_var, m_var, big_mu, big_sigma, r_w, m_w, fr, r_s, m_s = model(
                machine_labels_batch,
                rule_mus_batch,
                machine_mus_batch,
                rule_feature_activate_batch,
                machine_feature_activate_batch,
                machine_one_batch,
                y_risk_batch,
                y_mul_risk_batch,
                init_rule_mu,
                init_machine_mu,
                label_embeddings_batch  # Pass label embeddings
            )

            outputs_mul_all = torch.cat((outputs_mul_all.to(device[0]), torch.reshape(outputs_mul, (-1, class_num))), dim=0)
            outputs_all = torch.cat((outputs_all.to(device[0]), outputs), dim=0)

            # Update correct predictions calculations
            num_samples = len(test.true_labels)
            true_labels_batch = [test.true_labels[idx] for idx in range(bs * i, min(bs * i + bs, num_samples))]
            true_labels_tensor = torch.tensor(true_labels_batch, dtype=torch.float).to(device[0])
            
            # Make sure all tensors have the same number of labels
            label_num = true_labels_tensor.shape[1]  # Get the correct number of labels
            

        torch.cuda.empty_cache()
    
    outputs_mul_all = convert_tensor(outputs_mul_all)
    if (np.isnan(outputs_mul_all.reshape((-1)).cpu().numpy()).any()):
        print(outputs_mul_all.reshape((-1)).cpu().numpy())
    else:
        print("Fale")
    
    test.risk_mul_labels = np.array(test.risk_mul_labels)
    outputs_mul_all = outputs_mul_all.cpu().numpy()
    test_risk_mul_labels = test.risk_mul_labels[:, :, 1]
    outputs_mul_all = outputs_mul_all[:, :, 1]

    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(test_risk_mul_labels.shape[1]):
        labels = test_risk_mul_labels[:, i]
        preds = outputs_mul_all[:, i]
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
    #print('risk_mul_roc_auc')
    #print(roc_aucs)
    risk_mul_roc_auc = np.mean(roc_aucs)

    test.machine_mul_probs = np.array(test.machine_mul_probs)
    _machine_mul_pro = 1 - test.machine_mul_probs
    num_classes = test.risk_mul_labels.shape[2]
    num_labels = test.risk_mul_labels.shape[1]

    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(num_classes):
        for j in range(num_labels):
            labels = test.risk_mul_labels[:, j, i].flatten()
            preds = _machine_mul_pro[:, i * num_labels + j].flatten()
            fpr, tpr, _ = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            roc_aucs.append(roc_auc)
    #print('baseline_mul_roc_auc')
    #print(roc_aucs)
    baseline_mul_roc_auc = np.mean(roc_aucs)

    test_risk_labels = np.array(test.risk_labels)
    outputs_all = outputs_all.cpu().numpy()
    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(test_risk_labels.shape[1]):
        fpr, tpr, _ = roc_curve(test_risk_labels[:, i], outputs_all[:, i])
        roc_auc = auc(fpr, tpr) * 100
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
    #print('risk_roc_auc')
    print(roc_aucs)
    risk_roc_auc = np.mean(roc_aucs)

    _machine_pro = 1 - test.machine_probs
    test_risk_labels = np.array(test.risk_labels)
    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(test_risk_labels.shape[1]):
        fpr, tpr, _ = roc_curve(test_risk_labels[:, i], _machine_pro[:, i])
        roc_auc = auc(fpr, tpr) * 100
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
    
    baseline_roc_auc = np.mean(roc_aucs)

    
    logging.info("risk roc : %f,  baseline roc %f," % (risk_roc_auc, baseline_roc_auc))
   
    return outputs_all


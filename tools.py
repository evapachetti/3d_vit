# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import random
import numpy as np # type: ignore
import torch # type: ignore
import ml_collections # type: ignore
import pandas as pd # type: ignore

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize(dataset, mean, std=1):
    normalized_dataset = []
    
    for item in dataset:
        normalized_value = (item[0] - mean) / std
        normalized_item = (normalized_value,) + item[1:]
        normalized_dataset.append(normalized_item)
        
    return normalized_dataset


def save_best_metrics(save_path, model, specificity, sensitivity, accuracy, roc_auc, pr_auc, f2_score, true_labels, predicted_labels, class_probabilities):
    # Save the model state dictionary
    torch.save(model.state_dict(), save_path)
    
    # Return the metrics and labels directly
    return (specificity, sensitivity, accuracy, roc_auc, pr_auc, f2_score, true_labels, predicted_labels, class_probabilities)



def testing_model(dataloader, model, device):
    predicted_labels, true_labels, class_probabilities, features_vectors, logits = [],[],[],[],[]

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            y = y.long()
            
            output = model(x)
            logits.extend(output.tolist())
            p = torch.sigmoid(output)
            predicted = (p > 0.5).long()
            
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(y.tolist())
            class_probabilities.extend(p.tolist())

    return true_labels, predicted_labels, class_probabilities, features_vectors


def parameters_config(conf):
    configurations = {}
    
    # Define the configuration sets
    config_sets = [
        (range(1, 19), [2048, 3072], [4, 6, 8], [(64, 4), (32, 8), (16, 16)]),
        (range(19, 27), [2204], [4, 6], [(16, 4), (8, 8)])
    ]
    
    # Generate configurations
    for indices, mlp_dims, n_layers, hs_nh_pairs in config_sets:
        for k in indices:
            for dim in mlp_dims:
                for n in n_layers:
                    for hs, nh in hs_nh_pairs:
                        configurations[f'Configuration {k}'] = [8 if k >= 19 else 16, dim, n, hs, nh]

    # Retrieve the specified configuration
    ps, dim, n, hs, nh = configurations[f'Configuration {conf}']

    return ps, dim, n, hs, nh



def get_config(ps,dim,n,hs,nh):
    """Returns the EvaViT configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (ps, ps, 5)})
    config.hidden_size = hs
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = dim
    config.transformer.num_heads = nh
    config.transformer.num_layers = n
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def calculate_confidence_metrics(true_labels, predicted_labels, class_probabilities):

    true_negatives, true_negatives_0_3, false_negatives, true_positives, true_positives_0_7, false_positives = 0,0,0,0,0,0

    for true_label, predicted_label, class_prob in zip(true_labels, predicted_labels, class_probabilities):
        if true_label == 0 and predicted_label == 0:
            true_negatives += 1
            if class_prob < 0.3:
                true_negatives_0_3 += 1
        elif true_label == 1 and predicted_label == 0:
            false_negatives += 1
        elif true_label == 1 and predicted_label == 1:
            true_positives += 1
            if class_prob > 0.7:
                true_positives_0_7 += 1
        elif true_label == 0 and predicted_label == 1:
            false_positives += 1

    csp = true_negatives_0_3 / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    cse = true_positives_0_7 / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return csp, cse



def brier_score_one_class(y_true, y_prob, cl=0):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    mask = (y_true == cl)
    y_true_cl = y_true[mask]
    y_prob_cl = y_prob[mask]
    
    return np.average((y_true_cl - y_prob_cl) ** 2)



def bootstrapping(input_csv, seed):
    # Read the CSV file
    ind_data = pd.read_csv(input_csv)
    
    # Reset the index to ensure proper sampling
    ind_data.reset_index(drop=True, inplace=True)
    
    # Sample with replacement using the specified seed
    bootstrap_sample = ind_data.sample(n=len(ind_data), replace=True, random_state=seed)
    
    return bootstrap_sample




# def bootstrapping(input_csv, seed):
#     ind_data = pd.read_csv(input_csv) # lettura file csv dei dati di test
#     ind_data = ind_data.reset_index()
#     index = ind_data.index
#     ind_data_case_df = pd.DataFrame(list(ind_data['index']), columns=['Case'])
#     bootstrap_sample_size = len(ind_data_case_df)
#     bootstrap_sample = ind_data_case_df.sample(n = bootstrap_sample_size, replace = True, random_state = seed, axis='index') # campiono casualmente e con ripetizione le lesioni
#     new_indices = []
#     for j in bootstrap_sample.Case.tolist():
#         new_indices.append(index[ind_data['index'] == j].tolist())# prendo le fette corrispondenti alle lesioni selezionate (anche ripetute)
#     flat_new_indices = [item for sublist in new_indices for item in sublist] # metto tutti gli indici in un unico vettore
#     boot_trainset = ind_data.iloc[flat_new_indices,:].drop('index',axis=1)
      
#     return boot_trainset
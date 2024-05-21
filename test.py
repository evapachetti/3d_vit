
# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

from __future__ import absolute_import, division, print_function
import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
from sklearn.metrics import recall_score, roc_auc_score, fbeta_score, accuracy_score # type: ignore
from models.modeling import VisionTransformer
import numpy as np # type: ignore
import os
from tools import normalize, parameters_config, get_config, testing_model
from create_dataset import ProstateDataset, ToTensorDataset
from tqdm import tqdm # type: ignore
import argparse
import logging

def test(args):
    
    model = VisionTransformer(get_config(*parameters_config(args.conf)),args.image_size, zero_head=False, num_classes=1, vis=True)

    # Initialize results dictionaries
    results = {}
   
    for k in tqdm(range(args.cv)):
        # Load model
        model_path = os.path.join(args.base_path, f"conf{args.conf}", f"cv{k+1}.bin")
        model.load_state_dict(torch.load(model_path))
        model.to(args.device)
        model.eval()

        results[f'CV {k+1}'] = {}
        
        # Dataset loading
        csv_file_train = os.path.join(args.csv_path, f"training_cv{k+1}.csv")
        csv_file_val = os.path.join(args.csv_path, f"validation_cv{k+1}.csv")
        validset = ProstateDataset(csv_file_val)

        # Normalize data
        volumes_train = [i[0] for i in list(ProstateDataset(csv_file_train))]
        mean = np.mean(volumes_train)
        testset = normalize(testset, mean)
        validset = normalize(validset, mean)

        testset_tf = ToTensorDataset(testset, transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset_tf, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

        # Testing
        true_labels, predicted_labels, class_probabilities, _ = testing_model(testloader, model, args.device)
        accuracy = accuracy_score(true_labels, predicted_labels)
        specificity = recall_score(true_labels, predicted_labels, pos_label=0)
        sensitivity = recall_score(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, class_probabilities)
        f2_score = fbeta_score(true_labels, predicted_labels, beta=2)

        # Save results

        results[f'CV {k+1}']['Specificity'] = specificity
        results[f'CV {k+1}']['Sensitivity'] = sensitivity
        results[f'CV {k+1}']['Accuracy'] = accuracy
        results[f'CV {k+1}']['AUROC'] = roc_auc
        results[f'CV {k+1}']['F2-score'] = f2_score

    for k in tqdm(range(args.cv)):
        logging.info(f"Cross-validation fold: {k+1}")   
        for metric, value in results.items():
            logging.info(f"{metric}: {value}")   

    

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--cv", default = 5,
                        help="Number of folds in cross validation.")
    parser.add_argument("--conf", default = 5,
                        help="Configuration number of baseline model.")
    parser.add_argument("--max_configs", default = 19,
                        help="Max number of baseline configurations consider.")
    parser.add_argument("--combinations", default = 3,
                        help="How many baseline combinations in ensemble consider.")
    parser.add_argument("--image_size", default=128,
                        help="Image size.")
    parser.add_argument("--test_batch_size", default=1,
                        help="Batch size for validation and test loaders.")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to compute operations.")
    parser.add_argument("--csv_path", default=os.path.join(os.getcwd(), "csv_files", "cross_validation"),
                        help="Path where csv files are stored.")
    parser.add_argument("--output_path", default=os.path.join(os.getcwd(), "output"),
                        help="Path where store results.")
    parser.add_argument("--base_path", default=os.path.join(os.getcwd(), "output", "baseline_models"),
                        help="Path where baseline parameters are stored.")
    parser.add_argument("--ens_path", default=os.path.join(os.getcwd(), "output", "ensemble_models"),
                        help="Path where ensemble parameters are stored.")
    args = parser.parse_args()
    
    test(args)
   
if __name__ == "__main__":
    main()




import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import DamageDataset
#from model import EnhancedDamageModel
#from loss import adaptive_texture_loss
#import metrics  # Import the entire module
#from metrics import compute_ordinal_conf_matrix, calculate_xview2_score, print_f1_per_class, print_precision_per_class, print_recall_per_class
from utils import get_class_weights, analyze_class_distribution
#from visuals import plot_loss_curves, plot_multiclass_roc, visualize_predictions, plot_epoch_accuracy, plot_epoch_f1
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
from torchvision import transforms

try:
    from smote_func import smote_func, alt_func
except Exception as e:
    print(e)


def mkdir_results():
    os.makedirs('./results', exist_ok=True)  # create output directory
    highest = -1
    for folder in os.listdir('./results',):  # inside directory
        if int(folder) > highest:
            highest = int(folder)
    highest += 1
    return os.path.join('./results', f'{highest}')

def train_and_eval(use_glcm, patch_size, stride, batch_size, epochs, lr, root):
    """
    This file is modified from the original located in another repo
    """
    results_path = mkdir_results()                      # works to place contents of each individual run into respective directory
    os.makedirs(results_path, exist_ok=True)            # create output directory

    params = use_glcm, patch_size, stride, batch_size, epochs, lr, root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset paths
    train_pre = os.path.join(root, "img_pre")
    train_post = os.path.join(root, "img_post")
    train_mask = os.path.join(root, "gt_post")

    # Load dataset with patch size and stride (NOTE: only reads GT data)
    dataset = DamageDataset(train_pre, train_post, train_mask, patch_size=patch_size, stride=stride)
    print(f'{len(dataset)} samples to be scanned')
    # analyze_class_distribution(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("Begin SMOTE function...")
    # --------------------------
    #alt_func(train_dataset)
    smote_func(train_dataset, patch_size=patch_size)
    # --------------------------


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

train_and_eval(False, 64, 32, 4, 2, 16, "./data")

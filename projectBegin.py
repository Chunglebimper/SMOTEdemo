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



"""
Traditionally, SMOTE is used on binary data 
https://www.youtube.com/watch?v=U3X98xZ4_no
The problem with our data is that we need to apply SMOTE to indiviudal classes. This is a problem becuase we have to read the GT and identify areas with only the classes we want.
Do we feed in the patches with this data or the entire image?
"""
def mkdir_results():
    os.makedirs('./results', exist_ok=True)  # create output directory
    highest = -1
    for folder in os.listdir('./results',):  # inside directory
        if int(folder) > highest:
            highest = int(folder)
    highest += 1
    return os.path.join('./results', f'{highest}')


def SMOTE_func(X_train, Y_train, x_train, y_train, X_test, Y_test, target):
    """
    This function is meant to read in the class distribution and not augment data yet
    :param X_train:
    :param Y_train:
    :param x_train:
    :param y_train:
    :param X_test:
    :param Y_test:
    :return:
    """
    unique, count = np.unique(Y_train, return_counts=True)
    Y_train_dict_value_count = { k:v for (k,v) in zip(unique, count)}
    print("Before SMOTE patches: " + Y_train_dict_value_count)

    sm = SMOTE(random_state=12, ratio=1.0)              # key values to change
    x_train_res, y_train_res = sm.fit_sample(X_train, Y_train)

    unique, count = np.unique(y_train, return_counts=True)
    y_train_smote_value_count = { k:v for (k, v) in zip(unique, count)}
    print("After SMOTE patches: " + y_train_smote_value_count)

    clf = LogisticRegression().fit(x_train_res, y_train_res)
    Y_Test_Pred = clf.predict(X_test)

    pd.crosstab(pd.Series(Y_Test_Pred, name = 'Predicted'),
                pd.Series(Y_test[target], name = 'Actual' ))



def train_and_eval(use_glcm, patch_size, stride, batch_size, epochs, lr, root):
    results_path = mkdir_results()                      # works to place contents of each individual run into respective directory
    os.makedirs(results_path, exist_ok=True)            # create output directory

    params = use_glcm, patch_size, stride, batch_size, epochs, lr, root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset paths
    train_pre = os.path.join(root, "img_pre")
    train_post = os.path.join(root, "img_post")
    train_mask = os.path.join(root, "gt_post")

    # Load dataset with patch size and stride
    dataset = DamageDataset(train_pre, train_post, train_mask, patch_size=patch_size, stride=stride)
    #analyze_class_distribution(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #EXTRACT PATCHES FOR SMOTE:
    pre_patch, post_patch, mask_patch, patch_id = dataset[0]
    print(pre_patch, post_patch, mask_patch, patch_id)

train_and_eval(False, 64, 32, 4, 2, 16, "./data")
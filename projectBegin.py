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

# essential ph 1 phone

"""
Traditionally, SMOTE is used on binary data 
https://www.youtube.com/watch?v=U3X98xZ4_no
The problem with our data is that we need to apply SMOTE to indiviudal classes. This is a problem becuase we have to read the GT and identify areas with only the classes we want.
Furthermore, we do not have binary data (True/False type) we have 0-4 (5 types)


1. FEED THE PATCHES TO SMOTE WHERE THERE ARE CLASSES 3, 4; be sure to seperate training from validation
2. Find a way to get the (3,64,64) to be read as one value: 
    * Consider RGB to hex: (3, 64, 64) where the first dimension, 3, is roled into 
    * Justification: X_train and Y_train must be the same dimensions 

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
    :param X_train: Matrix containing the data which have to be sampled.
    :param Y_train: Corresponding label for each sample in X.
    :param x_train:
    :param y_train:
    :param X_test:
    :param Y_test:
    :return:
    """
    #X_train = X_train.reshape(X_train.shape[0], -1) #flatten posts
    X_train = X_train.flatten()
    print(Y_train)
    Y_train = Y_train.flatten()
    print(f"Shapes:\t {X_train.shape()}\n\t{Y_train.shape()}")
    unique, count = np.unique(Y_train, return_counts=True)
    Y_train_dict_value_count = { k:v for (k,v) in zip(unique, count)}
    print("Before SMOTE patches: " + str(Y_train_dict_value_count))

    sm = SMOTE(random_state=12)              # key values to change
    x_train_res, y_train_res = sm.fit_resample(X_train, Y_train)  # synthesized data:

    unique, count = np.unique(y_train, return_counts=True)
    y_train_smote_value_count = { k:v for (k, v) in zip(unique, count)}
    print("After SMOTE patches: " + str(y_train_smote_value_count))

    clf = LogisticRegression().fit(x_train_res, y_train_res)
    Y_Test_Pred = clf.predict(X_test)

    pd.crosstab(pd.Series(Y_Test_Pred, name = 'Predicted'),
                pd.Series(Y_test[target], name = 'Actual' ))



def train_and_eval(use_glcm, patch_size, stride, batch_size, epochs, lr, root):
    """
    This file is modified from the original located in another repo
    :param use_glcm:
    :param patch_size:
    :param stride:
    :param batch_size:
    :param epochs:
    :param lr:
    :param root:
    :return:
    """
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
    print(f'{len(dataset)} samples to be scanned')  # how is it getting that many images?
    #analyze_class_distribution(dataset)

    # Splits the dataset 80% training, 20% validation
    # May need to change train_size to adjust for SMOTE synthesized data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #EXTRACT PATCHES FOR SMOTE:
    pre_patch, post_patch, mask_patch, patch_id = dataset[0]
    # for every sample i
        # dataset[i][2] extract mask
    #print(pre_patch, post_patch, mask_patch, patch_id)
    #print(dataset)
    print(100 * "-")
    print("Begin SMOTE function...")
    print("Creating mask array...")
    print(type(dataset))

    from_tensor_mask = []
    from_tensor_post = []
    for i, x in enumerate(dataset):
        mask = x[2]                                     # gives me mask for element as tensor
        post = x[1]                                     #          post image       as tensor
        temp_np_array_mask = mask.cpu().numpy()         # Tensor to numpy
        temp_np_array_post = post.cpu().numpy()
        from_tensor_mask.append(temp_np_array_mask)
        from_tensor_post.append(temp_np_array_post)
        print(f"Mask {i} shape: {temp_np_array_mask.shape}")
        print(f"Post {i} shape: {temp_np_array_post.shape}")
        #print(str(temp_np_array_post))


    dataset_masks_np = np.array(from_tensor_mask)
    dataset_posts_np = np.array(from_tensor_post)
    #print(dataset_masks_np)
    print("Mask array generated")


    SMOTE_func(dataset_posts_np, dataset_masks_np, dataset, dataset, dataset, dataset, dataset)


train_and_eval(False, 64, 32, 4, 2, 16, "./data")

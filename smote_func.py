import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import numpy as np
# ^^^^^^^^^^^^^^^^^ Original packages for smote_func ^^^^^^^^^^^^^^^^^
import pandas as pd
import numpy as np
import os
import sys
from shutil import copyfile
import os.path
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img
from PIL import Image
from sklearn.model_selection import train_test_split
from numpy import load
import matplotlib.pyplot as plt
imagegen = ImageDataGenerator()
import rasterio

def display(boolean, post_patch_tensor):
    if boolean:
        print(np.shape(post_patch_tensor))
        try:
            img_tensor = post_patch_tensor.permute(1, 2, 0)  # shape: (64, 64, 3)
        except Exception as e:
            print(e)

        plt.imshow(img_tensor)
        plt.axis('off')
        plt.show()

def rgb_to_hex(r,g,b):
    print(r,g,b)
    return f"#{hex(r)}{hex(g)}{hex(b)}"

def smote_func(training_data, patch_size):
    """
    :param X_train: Matrix containing the data which have to be sampled.
    :param Y_train: Corresponding label for each sample in X.
    :param x_train:
    :param y_train:
    :param X_test:
    :param Y_test:
    :return:
    """
    
    # X_train is of the form: pre_patch, post_patch, torch.from_numpy(mask_patch).long(), f"{basename}_x{x}_y{y}"

    patchNum = 0
    X_train = training_data[patchNum]
    post_patch_tensor = X_train[1]                       # shape: (3, 64, 64)
    mask_patch_tensor = X_train[2]                       # shape: (64, 64)
    display(True, post_patch_tensor)

    try:
        print("Begin band handling and conversion...")
        post_patch_np = post_patch_tensor.numpy()  # shape: (3, 64, 64)
        mask_patch_np = mask_patch_tensor.numpy()
        print(np.shape(post_patch_np))

        reds, greens, blues = post_patch_np[0][0], post_patch_np[1][0], post_patch_np[2][0]
        r,g,b = reds[0], greens[0], blues[0]
        print(r, g, b)
        x = rgb_to_hex(r,g,b)
        print(x)
    except Exception as e:
        print(e)


    #try:
        #mask_array = np.array([X_train[i][2].cpu().detach().numpy() for i in range(len(X_train))])
        #print("Detached")
        #print(np.shape(mask_array))
    #except Exception as e:
        #print(f"Check device for tensor: {e}")

    # This feeds in a single band
    post_patch_np_edit = np.delete(post_patch_np, 0, axis=0)
    post_patch_np_edit = np.delete(post_patch_np_edit, 0, axis=0)
    post_patch_np_edit = post_patch_np_edit.squeeze()
    print(f"Before changes: {np.shape(post_patch_np)}, After changes {np.shape(post_patch_np_edit)}")

    mask_patch_np_edit = mask_patch_np
    print(f"Before changes: {np.shape(mask_patch_np)}, After changes {np.shape(mask_patch_np_edit)}")









    mask_patch_np_edit = np.zeros(64)
    for i in range(6):
        mask_patch_np_edit[i+10] = 1

    unique, count = np.unique(mask_patch_np_edit, return_counts=True)
    Y_train_dict_value_count = {k: v for (k, v) in zip(unique, count)}
    print("Before SMOTE patches: " + str(Y_train_dict_value_count))




    sm = SMOTE(random_state=2)
    x_train_res, y_train_res = sm.fit_resample(post_patch_np_edit, mask_patch_np_edit)  # synthesized data:
    unique, count = np.unique(y_train_res, return_counts=True)
    y_train_smote_value_count = {k: v for (k, v) in zip(unique, count)}
    print("After SMOTE patches: " + str(y_train_smote_value_count))
    display(True, x_train_res)

"""
def alt_func(train_dataset):
    train_generator = imagegen.flow(, class_mode="categorical",
                                                   shuffle=False, batch_size=128, target_size=(512, 512), seed=42)


    x = np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])
    y = np.concatenate([train_generator.next()[2] for i in range(train_generator.__len__())])
    print(x.shape)
    print(y.shape)
    # Converting  our color images to a vector
    X_train = x.reshape(103, 512 * 512 * 3)

    # Apply SMOTE method
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=2)
    X_smote, y_smote = sm.fit_resample(X_train, y)

    # Retrieve the image and save it to drive. Here's an example for a single image
    Xsmote_img = X_smote.reshape(585, 512, 512, 3)
    pil_img = array_to_img(Xsmote_img[80] * 255)
    pil_img.save('/content/gdrive/My Drive/demo.jpg')

    # Save all images generated by the SMOTE method to the drive

    train_sep_dir = '/content/gdrive/My Drive/idrid/testfolder/'

    # Create a "testfolder" if it does not exist on the drive
    if not os.path.exists(train_sep_dir):
        os.mkdir(train_sep_dir)

    # This function return label name
    def get_key(val):
        for key, value in patho.items():
            if val == value:
                return key

    for i in range(len(Xsmote_img)):
        label = get_key(str(y_smote[i]))
        if not os.path.exists(train_sep_dir + str(label)):
            os.mkdir(train_sep_dir + str(label))
        pil_img = array_to_img(Xsmote_img[i] * 255)
        pil_img.save(train_sep_dir + str(label) + '/' + 'smote_' + str(i) + '.jpg')
"""
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import numpy as np

def rgb_to_hex(r,g,b):
    print(r,g,b)
    return f"#{hex(r)}{hex(g)}{hex(b)}"

def smote_func(training_data):
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
    img_np = post_patch_tensor.permute(1, 2, 0).numpy()  # shape: (64, 64, 3)
    plt.imshow(img_np)
    plt.axis('off')
    #plt.show()

    print("Begin band handling and conversion...")
    post_patch_np = post_patch_tensor.numpy()  # shape: (3, 64, 64)
    mask_patch_np = mask_patch_tensor.numpy()
    print(np.shape(post_patch_np))

    reds, greens, blues = post_patch_np[0][0], post_patch_np[1][0], post_patch_np[2][0]
    r,g,b = reds[0], greens[0], blues[0]
    print(r, g, b)
    x = rgb_to_hex(r,g,b)
    print(x)


    """
    try:
        mask_array = np.array([X_train[i][2].cpu().detach().numpy() for i in range(len(X_train))])
        print("Detached")
        print(np.shape(mask_array))
    except Exception as e:
        print(f"Check device for tensor: {e}")
    """



    # from array of arrays? convert to an array

    sm = SMOTE(random_state=2)
    x_train_res, y_train_res = sm.fit_resample(post_patch_np, mask_patch_np)  # synthesized data:
    unique, count = np.unique(y_train, return_counts=True)
    #X_smote, y_smote = sm.fit_resample(X_train, y)
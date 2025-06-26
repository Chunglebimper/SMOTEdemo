import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import numpy as np

def smote_func(X_train, Y_train, x_train, y_train, X_test, Y_test, target):
    """
    :param X_train: Matrix containing the data which have to be sampled.
    :param Y_train: Corresponding label for each sample in X.
    :param x_train:
    :param y_train:
    :param X_test:
    :param Y_test:
    :return:
    """

    # get the post pieces of the training data;
    # shape torch[3,64,64]
    # X_train is of the form: pre_patch, post_patch, torch.from_numpy(mask_patch).long(), f"{basename}_x{x}_y{y}"
    for i in range(5):
        print(type(X_train[i][1]), X_train[i][1])
    # ---------------- LIST ----------------------------
    """
    post_patch_list = []
    for i in range(len(X_train)):
        _, post_patch, _, _ = X_train[i]
        post_patch_list.append(post_patch)
    """
    # ---------------- ARRAY ----------------------------
    img = X_train[0][1]  # shape: (3, 64, 64)
    img_np = img.permute(1, 2, 0).numpy()  # shape: (64, 64, 3)

    plt.imshow(img_np)
    plt.axis('off')
    plt.show()
    arr = np.ndarray(X_train[0][1][0])  # This is your tensor
    arr_ = np.squeeze(arr)  # you can give axis attribute if you wanna squeeze in specific dimension
    plt.imshow(arr_)
    plt.show()



    print("Begin band handling and conversion...")
    print(np.shape(X_train[0][1]))

    r, g, b = X_train[0][1][0][0], X_train[0][1][1][0], X_train[0][1][2][0]
    print(f'Red: {r}')
    if not all(0 <= x <= 255 for x in (r, g, b)):
        raise ValueError("RGB values must be between 0 and 255.")

    return f'#{r:02X}{g:02X}{b:02X}'



    print(X_train[0][1])
    post_patch_group = np.array([ (X_train[i][1]) for i in range(len(X_train))])

    # ---------------- TUPLE ----------------------------
    #post_patch_group = tuple([(X_train[i][1]) for i in range(len(X_train))])

    # the output of the line below is a numpy array
    try:
        mask_array = np.array([X_train[i][2].cpu().detach().numpy() for i in range(len(X_train))])
        print("Detached")
        print(np.shape(mask_array))
    except Exception as e:
        print(f"Check device for tensor: {e}")

    """ OLD WAY TO IMPLEMENT ABOVE LINE
    for i in range(len(X_train)):
        _, post_patch, _, _  = X_train[i]
        post_patch_list.append(post_patch)
    """
    print(post_patch_group)

    # from array of arrays? convert to an array

    sm = SMOTE(random_state=2)
    print(type(X_train))
    print(type(Y_train))
    x_train_res, y_train_res = sm.fit_resample(X_train, Y_train)  # synthesized data:
    unique, count = np.unique(y_train, return_counts=True)
    #X_smote, y_smote = sm.fit_resample(X_train, y)
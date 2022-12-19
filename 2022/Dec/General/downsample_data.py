import numpy as np
import random

def downsample_binary_data(X,y):
    # Indicies of each class' observations
    idx_inactives = np.where( y == 0 )[0]

    idx_actives   = np.where(y == 1 )[0]

    # Number of observations in each class
    num_inactives = len(idx_inactives)
    num_actives   = len(idx_actives)

    # Randomly sample from inactives without replacement
    np.random.seed(0)
    idx_inactives_downsampled = np.random.choice(idx_inactives, size=num_actives, replace=False)

    # Join together downsampled inactives with actives
    X = np.vstack((X[idx_inactives_downsampled], X[idx_actives]))
    X=np.concatenate((X[0], X[1]), axis=None)
    y = np.hstack((y[idx_inactives_downsampled], y[idx_actives]))
    #X,y=np.array(X),np.array(y)

    print("# inactives : ", len(y) - y.sum())
    print("# actives   : ", y.sum())
    print(len(X), len(y))
    return X,y

import numpy as np


def explained_variance(ypred, y, axis=-1):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    #assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y, axis=axis)
    return 1 - np.var(y-ypred, axis=axis)/vary

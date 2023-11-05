from sklearn.metrics import mean_absolute_error, median_absolute_error

import numpy as np


def Log(array):
    return np.log(1 + 0.002 * array)

def InvLog(array, std_err=0):
   return ((np.exp(array + ((std_err)**2) / 2) - 1) / 0.002)

def func_smape(A, F):
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0:  # Deals with a special case
        return 1
    return 1 / len_ * np.nansum(tmp)


def func_metrics(a, f):
    mae = mean_absolute_error(a, f)
    medae = median_absolute_error(a, f)
    smape = func_smape(a, f)
    rst = {
        "MAE": mae,
        "MedAE": medae,
        "smape": smape
    }
    return rst



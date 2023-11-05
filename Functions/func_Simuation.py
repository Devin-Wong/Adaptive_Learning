import numpy as np
import pandas as pd

from datetime import date

from pathlib import Path
import os
from Functions.func_smth_spl import spl

def func_number_errors(dat, dat_smooth, n_forecast=28):
    '''
    return:
        number of zeros in smooth errors
        number of non-zeros in smooth errors
        number of non-zeros for bootstrap errors
    '''

    e_u = np.array(dat - dat_smooth)
    
    i0 = 0  # obtain index with first non-zero

    for i in range(len(e_u)):
        if e_u[i] != 0:
            i0 = i
            break

    len_nonzero_e = len(e_u) - i0

    len_nonzero_e_bst = len_nonzero_e + n_forecast

    rst = {
        "zeros": i0,
        "non-zeros": len_nonzero_e,
        "bootstrap_non-zeros_errors": len_nonzero_e_bst,
        "e_u": e_u
    }

    return rst


def func_dataForSimuation(dat, dat_smooth, FIX_PT, p, path_data, n_blocks=10, N_sample=1000, n_forecast=28):
    
    N = len(dat)

    errs = func_number_errors(dat, dat_smooth)
    e_u = errs["e_u"]
    i0 = errs["zeros"]
    len_nonzero_e_bst = errs["bootstrap_non-zeros_errors"]
    len_block = int(len_nonzero_e_bst/n_blocks)

    e_u_0 = e_u[:(len(e_u)+n_forecast-len_block*n_blocks)]
    e_u_neq = e_u[i0:]

    np.random.seed(1014)
    starts = np.empty((N_sample, n_blocks))
    for i in range(N_sample):
        starts[i] = np.random.choice(
            int(len(e_u_neq) - len_block), n_blocks, replace=False)

    starts = starts.astype('int32')

    sample_e = np.empty((N_sample, n_blocks*len_block))

    for j in range(N_sample):
        start = starts[j]
        eee = np.array([])

        for i in range(n_blocks):
            eee = np.append(eee, np.array(e_u_neq[start[i]:(start[i]+len_block)]))

        sample_e[j] = eee


    sample_e = sample_e[:, -len_nonzero_e_bst:]

    # add zeros at the beginning
    E_0 = np.broadcast_to(e_u_0, (N_sample, len(e_u_0)))  # zeros

    sample_e = np.concatenate((E_0, sample_e), axis=1)
    
    sample_e_model = sample_e[:, :-n_forecast]
    sample_e_forecast = sample_e[:, -n_forecast:]


    dat_sample = np.empty(sample_e_model.shape)
    for i in range(len(dat_sample)):
        tem = dat_smooth + sample_e_model[i]
        # tem[-28:] = dat_log_model[-28:]  # keep the last 28 day no change
        dat_sample[i] = tem    

    # smooth 
    dat_sample_Smooth = []

    # i1 = 53
    for i in range(len(dat_sample)):
        yy = np.array(dat_sample[i])

        # spl_rst = spl(yy, i1=i1, p=p_v, fixEnd=1)
        spl_rst = spl(yy,  p, fixEndPoint=FIX_PT, i1=53, plot=False)
        spl_rst[spl_rst < 0] = 0

        dat_sample_Smooth.append(spl_rst)


    # save
    dat_sample = pd.DataFrame(dat_sample).T
    dat_sample_Smooth = pd.DataFrame(dat_sample_Smooth).T
# print(dat_sample.shape)


# ## save
    today = str(date.today())
    path_data = os.path.join(path_data, today)
    # print(path_data)
    Path(path_data).mkdir(parents=True, exist_ok=True)


    nm_tem = str(N) + "_p" + str(p)

    # ## sample data to be used in model
    file_name_sp = nm_tem + '_dat_sample' + '.csv'
    path_sp = os.path.join(path_data, file_name_sp)
    dat_sample.to_csv(path_sp, index=False)


    # ## sample smooth data to be used in model
    file_name_sed = nm_tem + '_dat_sample_Smooth' + '.csv'
    path_sed = os.path.join(path_data, file_name_sed)
    dat_sample_Smooth.to_csv(path_sed, index=False)
    # print(dat_sample_Smooth.shape)

    # # ## sample errors in model
    sample_e_model_df = pd.DataFrame(sample_e_model.T)
    file_name_se = nm_tem + "_sample_e_model" + '.csv'
    path_se = os.path.join(path_data, file_name_se)
    sample_e_model_df.to_csv(path_se, index=False)

    ## sample errors for forecasting
    sample_e_forecast_df = pd.DataFrame(sample_e_forecast.T)
    file_name_sf = nm_tem + '_sample_e_forecast'+'.csv'
    path_sf = os.path.join(path_data, file_name_sf)
    sample_e_forecast_df.to_csv(path_sf, index=False)

    # rst = {
    #     "dat_sample": dat_sample,
    #     "sample_e_model": sample_e_model,
    #     "sample_e_forecast": sample_e_forecast
    # }
    # return rst

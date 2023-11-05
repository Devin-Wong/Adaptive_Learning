from numpy.ma.core import std

from Functions.packages_basics import *

from Functions.func_smth_fixEnd import func_getFixPointByRLM
from Functions.func_smth_spl import spl, func_spl_mse, func_getSmoothLine
from Functions.func_smth_spl import func_spl_mse_any, func_getSmoothLine_any, spl_fixAny
from Functions.func_files import func_save_data
from Functions.settings import *

## settings -------------
n = 2
locs = [408, 415, 422]
N = locs[n]

ps_o = [16, 14, 14]
p_o = ps_o[n]
nm_tem_o = str(N) + "_p" + str(p_o)

n_forecast = 28

N1 = N + n_forecast

## 1. read log data-----------------------
dat_log_model = dat_log[:N]
dat_log_all = dat_log[:N1]


## 2. set fixed point
Fix_ends = [2.109, 2.158, 2.229, 2.327, 2.315, 2.171]
Fix_end = Fix_ends[n]

## 3. plot smth mse to select degrees of freedom
# smth_mses = func_spl_mse(dat_log_all, fixEndPoint=Fix_end, savePath=path_fig)
# smth_mses = func_spl_mse_any(dat_log_all, index_fix=N-1, savePath=path_fig, i0=53)

ps = [17, 15, 15, 14, 14, 16]
p = ps[n]

## 4. smooth
nm_tem = str(N1) + "_p" + str(p)

# dat_sm = func_getSmoothLine_any(dat_log_all, N1, p, index_fix=N-1,
#                        i0=53, savePath_fig=path_fig, savePath_data=path_dataSmooth)
dat_sm = pd.read_csv(os.path.join(path_dataSmooth, "forCheck",
                     nm_tem + "_smooth.csv"), index_col=0).iloc[:, 0].to_numpy()

# # ## 5. smooth + error

# 5.1 read error in model part
error_model = pd.read_csv(os.path.join(
    path_sim_data,  str(N), nm_tem_o+"_sample_e_model.csv"))

# 5.2 read error in forecast part
error_forecast = pd.read_csv(os.path.join(
    path_sim_data,  str(N), nm_tem_o+"_sample_e_forecast.csv"))

# print(error_model.shape)
# print(error_forecast.shape)    

# 5.3 get the sample data: smooth + error

dat_sample = np.empty([1000, N1])

dat_sample_smth = np.empty([1000, N1])
for i in range(1000):
    e1 = error_model.iloc[:,i].to_numpy()
    e2 = error_forecast.iloc[:, i].to_numpy()
    e = np.append(e1, e2)
    tem = dat_sm + e
    
    dat_sample[i] = tem

    # smooth sample data
    tem[N-1] = Fix_end
    sample_smth = spl_fixAny(tem,  p, index_fix=N-1, i1=53, plot=False)
    sample_smth[sample_smth<0]=0
    plt.plot(sample_smth)
    # print(sample_smth[N-1])
    dat_sample_smth[i] = sample_smth

plt.axvline(x=N-1)
plt.show()


dat_all = pd.DataFrame(dat_sample.T)
dat_all_smth = pd.DataFrame(dat_sample_smth.T)


# file_name = nm_tem_o + "_sample_all_0823"
# func_save_data(dat_all, file_name, path_sim_data)

# file_name1 = str(N) + "_model_forecast_sm_0823"
# func_save_data(dat_all_smth, file_name1, path_sim_forecast)

## 6. smooth data [:N] in the sample time-series

file_path = os.path.join(path_sim_data,str(N), nm_tem_o + "_sample_all_0823.csv")
dat_sample_all = pd.read_csv(file_path,index_col=0)
dat_sample_all = dat_sample_all.iloc[:N,:]

dat_sample_model_smth = np.empty([1000, N])
for i in range(1000):
    d = dat_sample_all.iloc[:,i].to_numpy().reshape(-1)
    d[-1] = Fix_end
    d_sm = spl_fixAny(d,  p_o, index_fix=N-1, i1=53, plot=False)
    d_sm[d_sm<0]=0
    dat_sample_model_smth[i] = d_sm
    
    plt.plot(d_sm)

plt.show()    
dat_sample_model_smth = pd.DataFrame(dat_sample_model_smth.T)

file_name2 = str(N) + "_dat_sample_Smooth_0826"
func_save_data(dat_sample_model_smth, file_name2, path_sim_data)

from numpy.ma.core import std

from Functions.packages_basics import *


from Functions.func_smth_fixEnd import func_getFixPointByRLM
from Functions.func_smth_spl import spl, func_spl_mse, func_getSmoothLine, func_spl_mse_any, spl_fixAny
from Functions.func_smth_spl import func_getSmoothLine_any
from Functions.settings import *

## -------- settings -------------
n = 0
locs = [408, 415, 422]
N = locs[n]

n_lastPoints = 28
n_forecast = 28

N_all = N + n_lastPoints

## 1. read log data-----------------------
# plt.plot(dat_log)
# plt.show()

# dat_log_model = dat_log[:N]
dat_log_model = dat_log[:(N+n_forecast)]

## 2. get fix end point

Fix_end = func_getFixPointByRLM(dat_log_model[:N], N, plot=True, savePath_fig=path_fig)
Fix_end = 2.109 # for 408

# ## 3. plot smth mse to select degrees of freedom
y = dat_log_model
y[-29] = Fix_end
smth_mses = func_spl_mse_any(y, index_fix=N-1, savePath=path_fig)


# ## 4. smooth 
# p = 17
# y = dat_log[:(N+n_forecast)]
# y[-29] = Fix_end
# index_fix = N-1
# len_smth = N+n_forecast

# # p = 16
# # y = dat_log
# # y[N-1] = Fix_end
# # index_fix = N-1
# # len_smth = N

# nm_tem = str(N) + "_p" + str(p)
# dat_sm = func_getSmoothLine_any(y, len_smth, p, index_fix=index_fix,
#                             i0=53, savePath_fig=path_fig, savePath_data=path_dataSmooth)                    
# # dat_sm = pd.read_csv(os.path.join(path_dataSmooth, nm_tem + "_smooth.csv"), index_col=0).iloc[:,0].to_numpy()

# ### s_y
# delta = dat_log[:N]-dat_sm
# s_y = std(delta)
# print(s_y)


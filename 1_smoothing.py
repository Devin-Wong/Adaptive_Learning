from numpy.ma.core import std

from Functions.packages_basics import *


from Functions.func_smth_fixEnd import func_getFixPointByRLM
from Functions.func_smth_spl import spl, func_spl_mse, func_getSmoothLine
from Functions.settings import *

## settings -------------
n = 0
locs = [408, 415, 422]
N = locs[n]

n_lastPoints = 28


## 1. read log data-----------------------
# plt.plot(dat_log)
# plt.show()

dat_log_model = dat_log[:N]

# print(len(dat.shape))
# plt.plot(dat_log)
# plt.axvline(x=N,color="red")
# plt.axvline(x=N+28, color="black")
# plt.show()

## 2. get fix end point
Fix_end = func_getFixPointByRLM(dat_log_model, N, plot=True, savePath_fig=path_fig)
# Fix_end = 2.109 # for 408


# # ## 3. plot smth mse to select degrees of freedom
smth_mses = func_spl_mse(dat_log_model, fixEndPoint=Fix_end, savePath=path_fig)

# ## 4. smooth 
p = 14
nm_tem = str(N) + "_p" + str(p)
dat_sm = func_getSmoothLine(dat_log, N, p, fixEndPoint=Fix_end,
                            i0=53, savePath_fig=path_fig, savePath_data=path_dataSmooth)                    
# # dat_sm = pd.read_csv(os.path.join(path_dataSmooth, nm_tem + "_smooth.csv"), index_col=0).iloc[:,0].to_numpy()

# ### s_y
# delta = dat_log[:N]-dat_sm
# s_y = std(delta)
# print(s_y)


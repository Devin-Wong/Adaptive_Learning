from numpy.ma.core import std
# from datetime import date
from Functions.packages_basics import *

# from Functions.func_smth_fixEnd import func_getFixPointByRLM
# from Functions.func_smth_spl import spl, func_spl_mse, func_getSmoothLine
from Functions.func_Simuation import func_number_errors
from Functions.func_Simuation import func_number_errors, func_dataForSimuation
from Functions.func_files import func_save_fig
from Functions.settings import *

## settings -------------
n = 0
locs = [408, 415, 422]
N = locs[n]

ps = [16, 14, 14]
p = ps[n]

n_lastPoints = 28
Fix_ends = [2.109, 2.158, 2.229]
Fix_end = Fix_ends[n]


nm_tem = str(N) + "_p" + str(p)

## 1. smoothing data
dat_sm = pd.read_csv(os.path.join(
    path_dataSmooth, nm_tem+"_smooth.csv"), index_col=0).iloc[:, 0].to_numpy().reshape(-1)
print(dat_sm)
# print(dat_sm)

## 5. data for smoothing
dat_log_model = dat_log[:N]
# error_number = func_number_errors(dat_log_model, dat_sm)
# print(error_number)

## 5. sim
# func_dataForSimuation(dat_log_model, dat_sm, Fix_end, p, path_sim_data)

dat_sample = pd.read_csv(os.path.join(
    path_sim_data, str(N), nm_tem+"_dat_sample_Smooth.csv"))

# print(dat_sm.head())    
dat_sample.plot(legend=None)
plt.title(nm_tem + ", Fix_end = " + str(Fix_end))
func_save_fig(nm_tem+"_sample_smooth_lines", path_fig)
plt.show()


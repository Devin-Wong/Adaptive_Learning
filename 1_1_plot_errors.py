import scipy.stats as stats
import pylab as py
import statsmodels.api as sm
from pathlib import Path
# from datetime import date
import plotly.graph_objects as go
from Functions.packages_basics import *
from Functions.settings import *
from Functions.func_files import func_getData_1Col, func_save_data, func_save_fig

from Functions.func_interval import func_pred_interval

n = 0
locs = [408, 415, 422]
N = locs[n]

ps = [16, 14, 14]
p = ps[n]
nm_tem = str(N) + "_p" + str(p)

## ----------- log data -------------------
dat_log = dat_log[:N]

## ----------------- smooth line -----------------------
file_name_smooth = nm_tem + "_smooth" + ".csv"
dat_smooth = func_getData_1Col(file_name_smooth, path_dataSmooth)

## errors
err = dat_log - dat_smooth
# plt.plot(err)
# plt.show()

# err_df = pd.DataFrame({str(N)+"_err": err})
# err_df.to_csv("data_sm_err/"+str(N)+"_err.csv")

## qq plot
stats.probplot(err[51:], dist="norm", plot=py)
py.show()

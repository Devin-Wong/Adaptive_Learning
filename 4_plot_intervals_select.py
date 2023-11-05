from torch import int16
from Functions.packages_basics import *
from Functions.settings import *
from Functions.func_files import func_getData_1Col, func_save_fig

from Functions.func_interval import func_pred_interval

n = 0
locs = [408, 415, 422]
N = locs[n]

ps = [16, 14, 14]
p = ps[n]
nm_tem = str(N) + "_p" + str(p)

suffix = "_noTransfer_loss"
## ----------------- orignal forecast ------------------------
file_name3 = nm_tem + "forecast" + ".csv"
dat_forecast3 = func_getData_1Col(file_name3, path_dataForecast)

## ----------------- smooth line -----------------------
file_name_smooth = nm_tem + "_smooth" + ".csv"
dat_smooth = func_getData_1Col(file_name_smooth, path_dataSmooth)

## ----------------- sample smooth lines ---------------------
file_name_sed = os.path.join(path_sim_data, str(N), nm_tem+'_dat_sample_Smooth' + '.csv')
dat_sample_Smooth = pd.read_csv(file_name_sed)

## ----------------- sim forecasts ---------------------------
# file_name_s = nm_tem + suffix + '.csv'
file_name_s = nm_tem + '_noTransfer_loss' + '.csv'
dat_fc = pd.read_csv(os.path.join(
    path_sim_forecast, file_name_s), header=None)

index_all = dat_fc.iloc[:, 0].astype(int).to_numpy().reshape(-1)
print(dat_fc.shape)

# ## ------------------ plot simulation lines ------------------

index_fc = dat_fc.iloc[:, 0].astype(int)

for i in index_all:
        fc = dat_fc[index_fc == i].to_numpy().reshape(-1)[1:]
        if fc[-1]<1.9:
            print(i)
            plt.plot(dat_sample_Smooth.iloc[:, i])
            plt.plot(range(N, N+28), fc)
# func_save_fig(nm_tem+"_sim_lines"+suffix,
#               os.path.join(path_fig, "simulation"))
plt.show()

# ## ------------------ plot intervals -----------------------
# fc_median, CI_U, CI_L, PI_U, PI_L = func_pred_interval(dat_fc.iloc[:,1:], ci=90)

# plt.rcParams["figure.figsize"] = (8, 8)
# range1 = range(408-14, len(dat_log))
# plt.plot(range1, dat_log[range1], color="black", linewidth=0.7)
# range2 = range(408-14, N)
# plt.plot(range2, dat_smooth[range2], color="black", label="smooth")

# plt.plot(range(N, N+28), fc_median, color="red", label="median")

# plt.plot(range(N, N+28), CI_U, color="green", label="90-CI")
# plt.plot(range(N, N+28), CI_L, color="green",  label="90-CI")

# plt.plot(range(N, N+28), PI_U, color="blue", label="90-PI")
# plt.plot(range(N, N+28), PI_L, color="blue",  label="90-PI")

# plt.rcParams["figure.figsize"] = (6, 6)
# plt.plot(range(N, N+28), dat_forecast3, color="brown", label="forecast")
# plt.title(nm_tem)
# plt.legend()
# func_save_fig(nm_tem+"_Intervals"+suffix, os.path.join(path_fig, "simulation"))
# plt.show()

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

suffix = "0_noTransfer"
## ----------------- orignal forecast ------------------------
file_name3 = nm_tem + "forecast" + ".csv"
dat_forecast3 = func_getData_1Col(file_name3, path_dataForecast)

## ----------------- smooth line -----------------------
file_name_smooth = nm_tem + "_smooth" + ".csv"
dat_smooth = func_getData_1Col(file_name_smooth, path_dataSmooth)

## ----------------- sample smooth lines ---------------------
file_name_sed = os.path.join(path_sim_data, str(N), nm_tem+'_dat_sample_Smooth' + '.csv')
dat_sample_Smooth = pd.read_csv(file_name_sed)
dat_sample_Smooth = dat_sample_Smooth.T

print(dat_sample_Smooth.head())

## ----------------- sim forecasts ---------------------------
file_name_l = nm_tem + '_noTransfer' + "_loss" + '.csv'
dat_fc_l = pd.read_csv(os.path.join(
    path_sim_forecast, file_name_l), header=None, index_col=0)
index_l = dat_fc_l.index.astype(int)
dat_fc_l = dat_fc_l.set_index(index_l)
print(index_l)
print(dat_fc_l.index)


## read sim forecasts in further 
file_name_vl = nm_tem + '_noTransfer' + "_val_loss" + '.csv'
dat_fc_vl = pd.read_csv(os.path.join(
    path_sim_forecast, file_name_vl), header=None, index_col=0)
index_vl = dat_fc_vl.index
print(index_vl)

## ------------------ plot simulation lines ------------------
# L = dat_fc.shape[0]
# ------ plot under fitting
## selected rows underfitted
dat_log_u = pd.read_csv("log/"+str(N)+"log_sel.csv")
indices_u = dat_log_u["index"].to_numpy()


# print rst by = loss
range1 = range(408-14,N)
count = 0
for i in index_l:
    if i not in indices_u:
        count += 1
        fc_l = dat_fc_l.loc[[i]].to_numpy().reshape(-1)

        plt.plot(range1, dat_sample_Smooth.iloc[i, (408-14):N])
        plt.plot(range(N, N+28), fc_l)
        plt.ylim((1, 2.7))
# func_save_fig(nm_tem+"_sim_lines"+suffix+"_underfitting", os.path.join(path_fig, "simulation"))
plt.show()

print(count)

# print rst by = loss
count = 0
for i in index_vl:
    if i not in indices_u:
        count += 1
        fc_vl = dat_fc_vl.loc[[i]].to_numpy().reshape(-1)

        plt.plot(range1, dat_sample_Smooth.iloc[i,(408-14):N])
        plt.plot(range(N, N+28), fc_vl)
        plt.ylim((1,2.7))
plt.show()
print(count)

# ## ------------------ plot intervals -----------------------
# CI = 80
# fc_median, CI_U, CI_L, PI_U, PI_L = func_pred_interval(dat_fc_vl, ci=CI)

# Intervals = {"CI_U": CI_U, "CI_L": CI_L, "PI_U": PI_U, "PI_L": PI_L}
# dat_Intervals = pd.DataFrame(Intervals)
# file_nm = nm_tem + "_interval_" + str(CI)
# func_save_data(dat_Intervals, file_nm, path_sim_forecast)


# plt.rcParams["figure.figsize"] = (8, 8)
# range1 = range(408-14, len(dat_log))
# plt.plot(range1, dat_log[range1], color="black", linewidth=0.7)
# range2 = range(408-14, N)
# plt.plot(range2, dat_smooth[range2], color="black", label="smooth")

# plt.plot(range(N, N+28), fc_median, color="red", label="median")

# plt.plot(range(N, N+28), CI_U, color="green", label=str(CI)+"-CI")
# plt.plot(range(N, N+28), CI_L, color="green",  label=str(CI)+"-CI")

# plt.plot(range(N, N+28), PI_U, color="blue", label=str(CI)+"-PI")
# plt.plot(range(N, N+28), PI_L, color="blue",  label=str(CI)+"-PI")

# plt.rcParams["figure.figsize"] = (6, 6)
# plt.plot(range(N, N+28), dat_forecast3, color="brown", label="forecast")
# plt.title(nm_tem)
# plt.legend()
# func_save_fig(nm_tem+"_Intervals"+suffix, os.path.join(path_fig, "simulation"))
# plt.show()

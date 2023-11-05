# from Functions.packages import *
from Functions.settings import *
from Functions.func_files import func_save_fig, func_getData_1Col

n = 0
locs = [408, 415, 422]
N = locs[n]

ps = [16, 14, 14]
p = ps[n]
nm_tem = str(N) + "_p" + str(p)


# ---------------------------------------------------------------
# raw log data being read in Functions/setting.py

# get smooth data
file_name_smooth = nm_tem + "_smooth" + ".csv"
dat_smooth = func_getData_1Col(file_name_smooth, path_dataSmooth)

# get forecast data
file_name = nm_tem + "forecast" + ".csv"
dat_forecast = func_getData_1Col(file_name, path_dataForecast)
# print(dat_forecast)
# get another forecast
file_name2 = nm_tem + "forecast" + "_4layers.csv"
dat_forecast2 = func_getData_1Col(file_name2, path_dataForecast)

## ------------------------ 1. plot all# ---------------------------------
N = len(dat_smooth)
plt.plot(dat_log, color="black", linewidth=0.7)
plt.plot(dat_smooth, color="blue")
plt.plot(range(N, N+28), dat_forecast, color="red", label="3 layers")
plt.plot(range(N, N+28), dat_forecast2, color="blue", label="4 layers")
plt.legend()
func_save_fig(nm_tem+"_forecast1", path_fig)
plt.show()

## ------------------------ 2. plot last parts ------------------------
plt.rcParams["figure.figsize"] = (8, 8)

N = len(dat_smooth)
NN0 = N - 14
range1 = range(NN0, len(dat_log))
plt.plot(range1, dat_log[range1], color="black", linewidth=0.7)
range2 = range(NN0, len(dat_smooth))
plt.plot(range2, dat_smooth[range2], color="blue")
plt.plot(range(N, N+28), dat_forecast, color="red", label="3 layers")
plt.plot(range(N, N+28), dat_forecast2, color="blue", label="4 layers")
# fig_name = nm_tem + "_forecast"
plt.legend()
func_save_fig(nm_tem + "_forecast_2", path_fig)
plt.show()















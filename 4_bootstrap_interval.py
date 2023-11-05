from pathlib import Path
import plotly.graph_objects as go
from Functions.packages_basics import *
from Functions.settings import *
from Functions.func_files import func_getData_1Col, func_save_data, func_save_fig

from Functions.func_interval import func_pred_interval

n = 2
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
file_name_sed = os.path.join(path_sim_data, str(
    N), nm_tem+'_dat_sample_Smooth' + '.csv')
dat_sample_Smooth = pd.read_csv(file_name_sed)
dat_sample_Smooth = dat_sample_Smooth.T

# print(dat_sample_Smooth.head())

## ----------------- sim forecasts ---------------------------
file_name_l = nm_tem + '_noTransfer' + "_loss" + ".csv"  # + '_0826.csv'

dat_fc_l = pd.read_csv(os.path.join(
    path_sim_forecast, file_name_l), header=None, index_col=0)
index_l = dat_fc_l.index.astype(int)


file_name_vl = nm_tem + '_noTransfer' + "_val_loss" + '.csv'
dat_fc_vl = pd.read_csv(os.path.join(
    path_sim_forecast, file_name_vl), header=None, index_col=0)
index_vl = dat_fc_vl.index.astype(int)

# print(index_vl)

### --------------------------
dat_fc_new = []
count = 0
# for i in num_wellfitting:
for i in index_l:
        fc = dat_fc_l.loc[[i]].to_numpy().reshape(-1)
      # print(fc)
        if fc[-1] < 1.8:
            fc = dat_fc_vl.loc[[i]].to_numpy().reshape(-1)
            count += 1
        dat_fc_new.append(fc)

#         plt.plot(dat_sample_Smooth.iloc[i, :])
#         plt.plot(range(N, N+28), fc)
#         # plt.ylim((1,2.7))
# plt.show()
# print(count)
dat_fc_new = pd.DataFrame(dat_fc_new)
print(dat_fc_new.shape)


## ----- bootstrap errors --------------
# read error in forecast part
# nm_tem_o = str(N) + "_p" + str(p_o)
error_forecast = pd.read_csv(os.path.join(
    path_sim_data,  str(N), nm_tem + "_sample_e_forecast.csv"))
# error_forecast.plot()
# plt.show()
print(error_forecast.shape)

## ------ add errors on forecasts  --------
dat_b = []
nrow, ncol = dat_fc_new.shape
for i in np.arange(nrow):
    v_fc = dat_fc_new.iloc[i,:]
    v_e = error_forecast.iloc[:,i]
    v_s = v_fc + v_e

    dat_b.append(v_s)

dat_b = pd.DataFrame(dat_b)
# dat_b.plot()
# plt.show()

## ------ prediction interval ---------
print(dat_b.shape)


def func_pred_interval(dat, ci=90):
  fc_median = dat.quantile(0.5, axis=0).to_numpy()
  fc_quantile_up = dat.quantile(1-(1-ci/100)/2, axis=0).to_numpy()
  fc_quantile_low = dat.quantile((1-ci/100)/2, axis=0).to_numpy()
  return fc_median, fc_quantile_low, fc_quantile_up

CI = 90
med, pi_l, pi_u = func_pred_interval(dat_b, CI)
print(med)
print(pi_l)
print(pi_u)

Intervals = {"PI_U": pi_l, "PI_L": pi_u, "median": med}
dat_Intervals = pd.DataFrame(Intervals)
file_nm = nm_tem + "_BootstrapInterval_" + str(CI)
func_save_data(dat_Intervals, file_nm, path_sim_forecast)

# -------- plot, using plotly -----------
N_all = 443+28

date = np.array(date)
layout = go.Layout(
    # title={
    #     'text': "Plot Title",
    #     'y': 0.9,
    #     'x': 0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top'},
    autosize=False,
    width=500,
    height=500,
    font=dict(
        size=11,
        color="black"
    ),
    xaxis_title="Date",
    yaxis_title="Shifted log transformantion",
)

fig = go.Figure(layout=layout)
N0 = 408-7
# log
# L_log = np.arange(N0, N_all)

range1 = np.arange(N0, 443+28)

N_forecast = 408+14+28
N_all = N_forecast + 14
range1 = np.arange(N0, N_all)

fig.add_trace(go.Scatter(x=date[range1], y=dat_log[range1],
                         mode='lines', line_width=1, line_color='black',
                         name='Log ')),

# smooth line
L_smooth = np.arange(N0, N)
fig.add_trace(go.Scatter(x=date[L_smooth], y=dat_smooth[L_smooth],
                         mode='lines', line_width=0.7, line_color="brown",
                         name="Smooth line")),

# prediction interval
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=pi_u,
                         mode='lines', line_width=1, line_color="rgb(127, 166, 238)"
                         )),
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=pi_l,
                         mode='lines', line_width=1, line_color="rgb(127, 166, 238)",
                         fill='tonexty'
                         )),

# median
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=med,
                         mode='lines', line_width=1, line_color="blue",
                         name="Median")),


fig.update_xaxes(
    # tick0= "27/2",
    # dtick= 7,  #"M1",
    # tickformat="%b\n%Y"
    tickformat="%m/%d",
)
fig.update_layout(showlegend=False)
fig.show()

# # save
# path_fig = os.path.join(path_fig, today)
# Path(path_fig).mkdir(parents=True, exist_ok=True)
# fig.write_image(path_fig + "/" + nm_tem + ".pdf")






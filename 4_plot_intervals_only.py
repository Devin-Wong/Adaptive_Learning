from pathlib import Path
# from datetime import date
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
file_name_sed = os.path.join(path_sim_data, str(N), nm_tem+'_dat_sample_Smooth' + '.csv')
dat_sample_Smooth = pd.read_csv(file_name_sed)
dat_sample_Smooth = dat_sample_Smooth.T.iloc[:, 1:]

CI = 90
## ------------------ read median ------------------
file_nm_m = nm_tem + "_median_" + str(CI) + ".csv"
dat_median = pd.read_csv(os.path.join(path_sim_forecast, "median", file_nm_m))

fc_median = dat_median["median"].to_numpy()
## ------------------ read intervals -----------------------


file_nm = nm_tem + "_interval_" + str(CI) + ".csv"
intervals = pd.read_csv(os.path.join(path_sim_forecast, "Intervals", file_nm))

CI_U = intervals["CI_U"].to_numpy()
CI_L = intervals["CI_L"].to_numpy()
PI_U = intervals["PI_U"].to_numpy()
PI_L = intervals["PI_L"].to_numpy()

# print(intervals.head())

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
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=PI_U,
                         mode='lines', line_width=1, line_color="rgb(127, 166, 238)"
                         )),
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=PI_L,
                         mode='lines', line_width=1, line_color="rgb(127, 166, 238)",
                         fill='tonexty'
                         )),

# confidence interval
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=CI_U,
                         mode='lines', line_width=1, line_color="rgb(131, 90, 241)"
                         )),
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=CI_L,
                         mode='lines', line_width=1, line_color="rgb(131, 90, 241)",
                         fill='tonexty'
                         )),


# median
fig.add_trace(go.Scatter(x=date[N:(N+28)], y=fc_median,
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

# save 
path_fig = os.path.join(path_fig, today)
Path(path_fig).mkdir(parents=True, exist_ok=True)

fig.write_image(path_fig + "/" + nm_tem + ".pdf")

from datetime import datetime, timedelta, date
from Functions.packages_basics import *

## --------- settings ------------------------
path_cwd = os.path.abspath(os.getcwd())

path_data = os.path.join(path_cwd, 'Data')
path_dataForecast = os.path.join(path_cwd, 'Data_forecast')
path_dataSmooth = os.path.join(path_cwd, 'Data_smooth')
path_fig = os.path.join(path_cwd, 'fig')
path_model = os.path.join(path_cwd, "Model")

path_sim_data = os.path.join(path_cwd, "Data_forSimulation")
path_sim_forecast = os.path.join(path_cwd, "Data_sim")

dat = pd.read_csv("Data/dat_log_noCorrection.csv", index_col=0)

dat_log = dat['dat_log'].to_numpy().reshape(-1)#[:(450+28)]


start_date = datetime(2020, 1, 23)
end_date = datetime(2021, 8, 24)


def date_range(start, end):
    delta = end - start  # as timedelta
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days


today = str(date.today())

date = date_range(start_date, end_date)
date = [str(d)[:10] for d in date]


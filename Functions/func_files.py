from pathlib import Path
import matplotlib.pyplot as plt
import os
from datetime import date, datetime
from Functions.packages_basics import *
## save images 

# path_cwd = os.path.abspath(os.getcwd())
# path_fig = os.path.join(path_cwd, 'fig')



def func_save_data(data, fig_name, path_data):
  today = str(date.today())
  path_data = os.path.join(path_data, today)
  Path(path_data).mkdir(parents=True, exist_ok=True)
  data.to_csv(os.path.join(path_data, fig_name + '.csv'))


def func_save_fig(fig_name, path_fig):
  now = datetime.now() # current date and time
  date_time = now.strftime("%m-%d-%Y %H-%M-%S")

  today = str(date.today())
  path_fig = os.path.join(path_fig, today)

  Path(path_fig).mkdir(parents=True, exist_ok=True)

  ss = fig_name + "-" + date_time + '.jpg'
  print(ss)
  plt.savefig(os.path.join(path_fig, ss))


def func_save_model(model, model_name, path_model):
  today = str(date.today())
  path_model = os.path.join(path_model, today)
  Path(path_model).mkdir(parents=True, exist_ok=True)
  nm = os.path.join(path_model, model_name+".h5")
  model.save(nm)


def func_getData_1Col(file_name, path):
    dat = pd.read_csv(os.path.join(path, file_name))
    if dat.shape[1] > 1:
        dat = dat.iloc[:, 1]
    dat = dat.to_numpy().reshape(-1)
    return dat

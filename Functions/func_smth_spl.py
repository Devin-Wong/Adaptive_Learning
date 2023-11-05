import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
# from Functions.func_files import func_save
import os
from pathlib import Path
from datetime import date

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def spl(y,  p, fixEndPoint=None, i1=0, plot=False):
    '''
    fixEndPoint: give the fixed end point.
    '''
    y = y[i1:]
    x = np.arange(len(y))
  #       x_origin = x
    if fixEndPoint is not None:
      y[-1] = fixEndPoint
      x0, y0 = x[-1], y[-1]
      x = x0 - x
      y = y - y0
    else:
      y0 = 0

    if type(p) is int:
      xx = np.quantile(x, np.arange(p) / (p - 1))
      xx = np.unique(xx)
      p = len(xx)
    else:
      xx = np.append(min(x), p)
      xx = np.append(xx, max(x))
      xx = np.unique(xx)
      p = len(xx)

    NN = len(x) * 10

    x_new = np.append(np.linspace(min(x), max(x), NN), x)

    z = np.array([x_new, np.power(x_new, 2)]).T

    i = 0
    for i in range(p - 1):
      x_new_1 = x_new - xx[i]
      bo = np.greater_equal(x_new, xx[i])
      col_new = np.power(bo * 1 * x_new_1, 3).reshape(-1, 1)
      z = np.append(z, col_new, axis=1)

    X1 = z[:-len(x), :]
    X2 = z[-len(x):, :]

  #       a, _, _, _ = np.linalg.lstsq(X2, y)
    if fixEndPoint is not None:
      a = np.linalg.lstsq(X2, y, rcond=None)[0]
      a = np.array(a).reshape(-1, 1)
      m = np.dot(X2, a)
    else:
      X21 = np.c_[X2, np.ones(len(X2))]
      a = np.linalg.lstsq(X21, y, rcond=None)[0]
      a = np.array(a).reshape(-1, 1)
      m = np.dot(X21, a)

    y_rst = m + y0

    y_rst = np.append(np.zeros(i1), y_rst)

    ### for plot more details ####
    if plot:
      x1_df = pd.DataFrame(X1)
      x2_df = pd.DataFrame(X2)
      if fixEndPoint is not None:
        m1 = np.dot(X1, a)
      else:
        X11 = np.c_[X1, np.ones(len(X1))]
        m1 = np.dot(X11, a)

      yyy = m1 + y0
  #           xxx = X1[:,0]
      xxx = np.append(np.linspace(0, i1 - 1, i1 * 10), X1[:, 0] + i1)
      plt.plot(np.append(np.zeros(i1), y + y0), label="real")

  #           plt.plot(x0-xxx,yyy, label="more fitting details")
      if fixEndPoint is not None:
        plt.plot(x0 - xxx, yyy, label="more fitting details")
      else:
        plt.plot(xxx, yyy, label="more fitting details")

      plt.plot(y_rst, label='fitting')
      plt.legend()
      plt.show()

    return y_rst


def func_spl_mse(y, fixEndPoint=None, savePath=None, i0=53):
  mses = []
  yy = y.copy()
  N = len(yy)

  for p in range(10, 30):
    if fixEndPoint is not None:
      # yy[-1] = fixEndPoint
      dat_smooth_t = spl(yy, p=p, fixEndPoint=fixEndPoint, i1=i0)
    else:
      dat_smooth_t = spl(yy, p=p, i1=i0)
    mse = mean_squared_error(dat_smooth_t, y)

    mses.append(mse)

  ## second difference
  # diff1 = diff(mses)

  plt.plot(range(10, 30), mses)
  plt.xticks(range(10, 30))

  plt.grid()
  plt.title(str(N)+", FixEnd="+str(fixEndPoint)[:5])
  plt.xlabel("Degree of freedom")
  plt.ylabel("Smooth splines MSE")
  
  if savePath is not None:
    today = str(date.today())
    savePath = os.path.join(savePath, today)

    Path(savePath).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(savePath, str(N) + "_smooth MSE.jpg"))
  
  plt.show()

  return mses


def func_getSmoothLine(y0, N, p, fixEndPoint=None, i0=53, savePath_fig=None, savePath_data=None):
  # N = len(y)
  y = y0[:N]
  plt.plot(y0)
  # y = dat.copy()

  if fixEndPoint is not None:
    # y[-1] = FIX_PT
    y_sm = spl(y, p=p, fixEndPoint=fixEndPoint, i1=i0)
    
    y_sm[y_sm < 0] = 0
  else:
    y_sm = spl(y, p=p, i1=i0)
    y_sm[y_sm < 0] = 0
  
  plt.plot(y_sm)

  if fixEndPoint is not None:
    title = str(N) + ", p=" + str(p) + ", FixEnd="+str(fixEndPoint)[:5]
  else:
    title = str(N) + ", p=" + str(p) + ", No fix last point"

  plt.title(title)
  
  if savePath_fig is not None:
    today = str(date.today())
    savePath_fig = os.path.join(savePath_fig, today)
    Path(savePath_fig).mkdir(parents=True, exist_ok=True)

    if fixEndPoint is not None:
      filename = str(N) + "p" + str(p) + "_FixEnd_smooth Line.jpg"
    else:
      filename = str(N) + "p" + str(p) + "_noFixEnd_smooth Line.jpg"

    plt.savefig(os.path.join(savePath_fig, filename))

  plt.show()

  if savePath_data is not None:
    today = str(date.today())
    savePath_data = os.path.join(savePath_data, today)

    dct_sm = {str(N)+"_p"+str(p): y_sm.reshape(-1)}
    dct_sm = pd.DataFrame(dct_sm)
    Path(savePath_data).mkdir(parents=True, exist_ok=True)
    nm_tem = str(N) + "_p" + str(p)
    dct_sm.to_csv(os.path.join(savePath_data, nm_tem + "_smooth.csv"))

  return y_sm


def func_getSmoothLine_any(y0,N, p, index_fix=None, i0=53, savePath_fig=None, savePath_data=None):
  # N = len(y)
  y = y0
  y = y[:N]
  plt.plot(y0)
  # y = dat.copy()

  if index_fix is not None:
    # y[-1] = FIX_PT
    y_sm = spl_fixAny(y, p=p, index_fix=index_fix, i1=i0)
    y_sm[y_sm < 0] = 0
  else:
    y_sm = spl(y, p=p, i1=i0)
    y_sm[y_sm < 0] = 0

  plt.plot(y_sm)

  if index_fix is not None:
    title = str(N) + ", p=" + str(p) + ", index_fix="+str(index_fix)
  else:
    title = str(N) + ", p=" + str(p) + ", No fix last point"

  plt.title(title)

  if savePath_fig is not None:
    today = str(date.today())
    savePath_fig = os.path.join(savePath_fig, today)
    Path(savePath_fig).mkdir(parents=True, exist_ok=True)

    if index_fix is not None:
      filename = str(N) + "p" + str(p) + "_FixEnd_smooth Line.jpg"
    else:
      filename = str(N) + "p" + str(p) + "_noFixEnd_smooth Line.jpg"

    plt.savefig(os.path.join(savePath_fig, filename))

  plt.show()

  if savePath_data is not None:
    today = str(date.today())
    savePath_data = os.path.join(savePath_data, today)

    dct_sm = {str(N)+"_p"+str(p): y_sm.reshape(-1)}
    dct_sm = pd.DataFrame(dct_sm)
    Path(savePath_data).mkdir(parents=True, exist_ok=True)
    nm_tem = str(N) + "_p" + str(p)
    dct_sm.to_csv(os.path.join(savePath_data, nm_tem + "_smooth.csv"))

  return y_sm

def spl(y,  p, fixEndPoint=None, i1=0, plot=False):
    '''
    fixEndPoint: give the fixed end point.
    '''
    y = y[i1:]
    x = np.arange(len(y))
  #       x_origin = x
    if fixEndPoint is not None:
      y[-1] = fixEndPoint
      x0, y0 = x[-1], y[-1]
      x = x0 - x
      y = y - y0
    else:
      y0 = 0

    if type(p) is int:
      xx = np.quantile(x, np.arange(p) / (p - 1))
      xx = np.unique(xx)
      p = len(xx)
    else:
      xx = np.append(min(x), p)
      xx = np.append(xx, max(x))
      xx = np.unique(xx)
      p = len(xx)

    NN = len(x) * 10

    x_new = np.append(np.linspace(min(x), max(x), NN), x)

    z = np.array([x_new, np.power(x_new, 2)]).T

    i = 0
    for i in range(p - 1):
      x_new_1 = x_new - xx[i]
      bo = np.greater_equal(x_new, xx[i])
      col_new = np.power(bo * 1 * x_new_1, 3).reshape(-1, 1)
      z = np.append(z, col_new, axis=1)

    X1 = z[:-len(x), :]
    X2 = z[-len(x):, :]

  #       a, _, _, _ = np.linalg.lstsq(X2, y)
    if fixEndPoint is not None:
      a = np.linalg.lstsq(X2, y, rcond=None)[0]
      a = np.array(a).reshape(-1, 1)
      m = np.dot(X2, a)
    else:
      X21 = np.c_[X2, np.ones(len(X2))]
      a = np.linalg.lstsq(X21, y, rcond=None)[0]
      a = np.array(a).reshape(-1, 1)
      m = np.dot(X21, a)

    y_rst = m + y0

    y_rst = np.append(np.zeros(i1), y_rst)

    ### for plot more details ####
    if plot:
      x1_df = pd.DataFrame(X1)
      x2_df = pd.DataFrame(X2)
      if fixEndPoint is not None:
        m1 = np.dot(X1, a)
      else:
        X11 = np.c_[X1, np.ones(len(X1))]
        m1 = np.dot(X11, a)

      yyy = m1 + y0
  #           xxx = X1[:,0]
      xxx = np.append(np.linspace(0, i1 - 1, i1 * 10), X1[:, 0] + i1)
      plt.plot(np.append(np.zeros(i1), y + y0), label="real")

  #           plt.plot(x0-xxx,yyy, label="more fitting details")
      if fixEndPoint is not None:
        plt.plot(x0 - xxx, yyy, label="more fitting details")
      else:
        plt.plot(xxx, yyy, label="more fitting details")

      plt.plot(y_rst, label='fitting')
      plt.legend()
      plt.show()

    return y_rst


def spl_fixAny(y,  p, index_fix=None, i1=0, plot=False):
    '''
    fixEndPoint: give the fixed end point.
    '''
    y = y[i1:]
    x = np.arange(len(y))
  #       x_origin = x
    index_fix = index_fix - i1
    if index_fix is not None:
      x0, y0 = x[index_fix], y[index_fix]
      x = x0 - x
      y = y - y0
    else:
      y0 = 0

    if type(p) is int:
      xx = np.quantile(x, np.arange(p) / (p - 1))
      xx = np.unique(xx)
      p = len(xx)
    else:
      xx = np.append(min(x), p)
      xx = np.append(xx, max(x))
      xx = np.unique(xx)
      p = len(xx)

    NN = len(x) * 10

    x_new = np.append(np.linspace(min(x), max(x), NN), x)

    z = np.array([x_new, np.power(x_new, 2)]).T
    
    i = 0
    for i in range(p - 1):
      x_new_1 = x_new - xx[i]
      bo = np.greater_equal(x_new, xx[i])
      col_new = np.power(bo * 1 * x_new_1, 3).reshape(-1, 1)
      z = np.append(z, col_new, axis=1)

    ## -----
    zero_row = z[NN+index_fix,:]
    z = np.subtract(z, zero_row)
    ## ----
    
    X1 = z[:-len(x), :]
    X2 = z[-len(x):, :]
  
  #       a, _, _, _ = np.linalg.lstsq(X2, y)
    if index_fix is not None:
      a = np.linalg.lstsq(X2, y, rcond=None)[0]
      a = np.array(a).reshape(-1, 1)
      m = np.dot(X2, a)

      # reg = LinearRegression(fit_intercept=False).fit(X2, y)
      # a = reg.coef_
      # # print(reg.intercept_)
      # # m = np.dot(X2, a)
      # m = reg.predict(X2)

    else:
      X21 = np.c_[X2, np.ones(len(X2))]
      a = np.linalg.lstsq(X21, y, rcond=None)[0]
      a = np.array(a).reshape(-1, 1)
      m = np.dot(X21, a)

    y_rst = m + y0

    y_rst = np.append(np.zeros(i1), y_rst)

    ### for plot more details ####
    if plot:
      x1_df = pd.DataFrame(X1)
      x2_df = pd.DataFrame(X2)
      if index_fix is not None:
        m1 = np.dot(X1, a)
      else:
        X11 = np.c_[X1, np.ones(len(X1))]
        m1 = np.dot(X11, a)

      yyy = m1 + y0
  #           xxx = X1[:,0]
      xxx = np.append(np.linspace(0, i1 - 1, i1 * 10), X1[:, 0] + i1)
      plt.plot(np.append(np.zeros(i1), y + y0), label="real")

  #           plt.plot(x0-xxx,yyy, label="more fitting details")
      if index_fix is not None:
        plt.plot(x0 - xxx, yyy, label="more fitting details")
      else:
        plt.plot(xxx, yyy, label="more fitting details")

      plt.plot(y_rst, label='fitting')
      plt.legend()
      plt.show()

    return y_rst

def func_spl_mse(y, fixEndPoint=None, savePath=None, i0=53):
  mses = []
  yy = y.copy()
  N = len(yy)

  for p in range(10, 30):
    if fixEndPoint is not None:
      # yy[-1] = fixEndPoint
      dat_smooth_t = spl(yy, p=p, fixEndPoint=fixEndPoint, i1=i0)
    else:
      dat_smooth_t = spl(yy, p=p, i1=i0)
    mse = mean_squared_error(dat_smooth_t, y)

    mses.append(mse)

  ## second difference
  # diff1 = diff(mses)

  plt.plot(range(10, 30), mses)
  plt.xticks(range(10, 30))

  plt.grid()
  plt.title(str(N)+", FixEnd="+str(fixEndPoint)[:5])
  plt.xlabel("Degree of freedom")
  plt.ylabel("Smooth splines MSE")

  if savePath is not None:
    today = str(date.today())
    savePath = os.path.join(savePath, today)

    Path(savePath).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(savePath, str(N) + "_smooth MSE.jpg"))

  plt.show()

  return mses


def func_spl_mse_any(y, index_fix=None, savePath=None, i0=53):
  mses = []
  yy = y.copy()
  N = len(yy)

  for p in range(10, 30):
    if index_fix is not None:
      # yy[-1] = fixEndPoint
      dat_smooth_t = spl_fixAny(yy, p=p, index_fix=index_fix, i1=i0)
    else:
      dat_smooth_t = spl(yy, p=p, i1=i0)
    mse = mean_squared_error(dat_smooth_t, y)

    mses.append(mse)

  ## second difference
  # diff1 = diff(mses)

  plt.plot(range(10, 30), mses)
  plt.xticks(range(10, 30))

  plt.grid()
  plt.title(str(N)+", Fix Index="+str(index_fix))
  plt.xlabel("Degree of freedom")
  plt.ylabel("Smooth splines MSE")

  if savePath is not None:
    today = str(date.today())
    savePath = os.path.join(savePath, today)

    Path(savePath).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(savePath, str(N) + "_smooth MSE.jpg"))

  plt.show()

  return mses

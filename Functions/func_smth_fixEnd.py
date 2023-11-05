import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import date

from pathlib import Path

from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
# import statsmodels as sm
import os

# from Functions.func_files import func_save


## get fix point
def func_getFixPointByRLM(y, loc, n_lastPoints=28, plot=False, savePath_fig=None):
    '''
    plot - True: plot; flase: no plot
    file_nm - Give a string value, will save the plot to /fig
    '''
    
    x1 = np.arange(n_lastPoints)
    x2 = x1**2
    y = y[-n_lastPoints:]

    D = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    rlm_fit = smf.rlm('y~x1+x2', data=D).fit()
    FIX_PT = rlm_fit.fittedvalues.to_numpy()[-1]

    if plot:
        plt.plot(y)
        plt.plot(rlm_fit.fittedvalues)

        plt.grid()
        plt.xlabel("the last 28 points")
        plt.ylabel("Shifted log transformation")
        plt.title(str(loc) + ", RLM, Last point = " + str(FIX_PT)[:5])
        if savePath_fig is not None:
            today = str(date.today())
            savePath_fig = os.path.join(savePath_fig, today)
            Path(savePath_fig).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(savePath_fig, str(loc)+"_RLM.jpg"))
        plt.show()

    return FIX_PT


def func_getFixPointByLinReg(pts, quadratic=False, plot=False):
  x = np.arange(len(pts)).reshape((len(pts), -1))

  if quadratic:
    x = np.arange(len(pts))
    x = np.vstack([x, x**2]).T

  reg = LinearRegression().fit(x, pts)
  pts_fit = reg.predict(x)

  if plot:
    plt.plot(pts)
    plt.plot(pts_fit)
    plt.show()

  return pts_fit[-1]



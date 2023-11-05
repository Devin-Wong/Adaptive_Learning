from Functions.packages_basics import *

def func_pred_interval(dat, s_y=0.158, ci=90):
  fc_median = dat.quantile(0.5, axis=0).to_numpy()
  fc_quantile_up = dat.quantile(1-(1-ci/100)/2, axis=0).to_numpy()
  fc_quantile_low = dat.quantile((1-ci/100)/2, axis=0).to_numpy()

  # diff = fc_quantile_up - fc_quantile_low

  #  ------------
  diff_u, diff_l = fc_quantile_up - fc_median, fc_median - fc_quantile_low
  diff_u_1, diff_l_1 = diff_u, diff_l
  
  for i in range(len(fc_median)-1):
    if diff_u[i+1]<diff_u[i]:
      diff_u_1[i+1] = diff_u_1[i]

    if diff_l[i+1] < diff_l[i]:
      diff_l_1[i+1] = diff_l_1[i]

  ConIntvU = fc_median + diff_u_1
  ConIntvL = fc_median - diff_l_1
  #  ------------

  Delta = 0.5 * (diff_u_1 + diff_l_1)
  # print(Delta)

  if ci == 90:
    t = 1.6448536
  elif ci == 80:
    t = 1.2815516
  elif ci == 95:
    t = 1.959964

  U = fc_median + np.sqrt((t*s_y)**2 + Delta**2)
  L = fc_median - np.sqrt((t*s_y)**2 + Delta**2)
  # print(L)
  # return {"median": fc_median, "CI_U":fc_95, "CI_L":fc_05, "PI_U": U, "PI_L": L}
  return fc_median, ConIntvU, ConIntvL, U, L


def func_pred_interval_0(dat, s_y=0.158, ci=90):
  fc_median = dat.quantile(0.5, axis=0)
  fc_quantile_up = dat.quantile(1-(1-ci/100)/2, axis=0).to_numpy()
  fc_quantile_low = dat.quantile((1-ci/100)/2, axis=0).to_numpy()

  diff = fc_quantile_up - fc_quantile_low

  Delta = 0.5 * diff
  # print(Delta)

  if ci == 90:
    t = 1.6448536
  elif ci == 80:
    t = 1.2815516
  elif ci == 95:
    t = 1.959964

  U = fc_median + np.sqrt((t*s_y)**2 + Delta**2)
  L = fc_median - np.sqrt((t*s_y)**2 + Delta**2)
  # print(L)
  # return {"median": fc_median, "CI_U":fc_95, "CI_L":fc_05, "PI_U": U, "PI_L": L}
  return fc_median, fc_quantile_up, fc_quantile_low, U, L

from Functions.packages_basics import *
from Functions.settings import *

## settings -------------
n = 0
locs = [408, 415, 422]
N = locs[n]

ps = [16, 14, 14]
p = ps[n]
nm_tem = str(N) + "_p" + str(p)

CI = 90

# 1. read Confidence interval
fl_path0 = os.path.join(path_sim_forecast, "Intervals1",
                        nm_tem+"_interval_"+str(CI) + ".csv")
dat_interval = pd.read_csv(fl_path0).iloc[:, 1:]
# print(dat_interval)
ci_u, ci_l = dat_interval["CI_U"].to_numpy(), dat_interval["CI_L"].to_numpy()

# 2. read smooth sample data with fix point [-29]
fl_path = os.path.join(path_sim_forecast, "Data_sample_smth",
                       str(N)+"_model_forecast_sm_0823.csv")
dat_sample_smth = pd.read_csv(fl_path).iloc[:,1:]
print(dat_sample_smth.shape)
nrow = dat_sample_smth.shape[0]
# range1 = np.arange(408-14, 443+28)-53
for i in range(1000):
    plt.plot(dat_sample_smth.iloc[-56:,i], color="gray")
plt.plot(np.arange((nrow-28),nrow),ci_u, color="red")
plt.plot(np.arange((nrow-28), nrow), ci_l, color="red")
plt.title(str(N))
plt.show()

dat_sample_smth_last28 = dat_sample_smth.iloc[-28:, :]
# print(dat_sample_smth_last28.head())

# 3. check interval coverage
L = []
s = 0
for l in range(dat_sample_smth_last28.shape[1]):
    d = dat_sample_smth_last28.iloc[:, l].to_numpy()

    b1, b2 = d < ci_u, d > ci_l
    l_c = []
    for i in range(len(b1)):
        c = (b1[i] and b2[i])*1
        # print(c)
        # if not c:
        #     print(f"{d[i]}, u={pi_u[i]}, l={pi_l[i]}")
        # l_c.append(c)

        s += c
    L.append(l_c)

# print(L)

print(s/(28*1000))

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
# 1. read intervals
# fl_path0 = os.path.join(path_sim_forecast, "Intervals1", nm_tem+"_interval_"+str(CI) +".csv")
# dat_interval = pd.read_csv(fl_path0).iloc[:,1:]
# # print(dat_interval)
# pi_u, pi_l = dat_interval["PI_U"].to_numpy(), dat_interval["PI_L"].to_numpy()

## read bootstrap confidence interval
fl_path0 = os.path.join(path_sim_forecast, "Intervals_bootstrap",
                        nm_tem+"_BootstrapInterval_"+str(CI) + ".csv")
print(fl_path0)

dat_interval = pd.read_csv(fl_path0).iloc[:, 1:]

pi_l, pi_u = dat_interval["PI_U"].to_numpy(), dat_interval["PI_L"].to_numpy()

print("pi_l")
print(pi_l)
print("pi_u")
print(pi_u)

# 2. read sample data
fl_path = os.path.join(path_sim_data, str(N), nm_tem+"_sample_all_0823.csv")
dat_sample = pd.read_csv(fl_path, index_col=0)
dat_sample_last28 = dat_sample.iloc[-28:, :]
print(dat_sample_last28.head())

# 3. judge whether sample data is in the interval

L = []
s = 0
for l in range(dat_sample_last28.shape[1]):
# for l in range(1):
    d = dat_sample_last28.iloc[:, l].to_numpy()
    
    # print(d)

    b1, b2 = d < pi_u, d > pi_l
    # print(b1)
    # print(b2)
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
print(s)
print(s/(28*1000))

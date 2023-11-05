from Functions.packages_basics import *

file = "log/408log"
log = pd.read_csv(file+".csv")

# select rows with val_los > 0.0025
log_underfitting = log[log["val_loss"]>0.0025]

log_underfitting.to_csv(file+"_sel1.csv", index=False)
# print(log_underfitting.head())


# get the row indices

# log = pd.read_csv("log/408log_sel.csv")

# indx = log["index"][log["val_loss"] > 0.0025].to_numpy()
# print(indx)

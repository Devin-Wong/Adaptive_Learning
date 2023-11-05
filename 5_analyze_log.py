from Functions.packages_basics import *

nm_tem = "408"
log_path = '/Users/Devin/Downloads/408nf1.log'

with open("log/"+nm_tem+"log.csv", 'a') as fl:
    write = csv.writer(fl)
    write.writerow(["index", "loss", "val_loss"])


def func_get_index(line):
    x = line.split(" ")[1][:-1]
    return int(x)

def func_get_loss(line):
    x = line.split("-")
    xloss, xval_loss = x[2], x[3]
    loss, val_loss = xloss.split(": ")[1], xval_loss.split(": ")[1]
    return float(loss), float(val_loss)

with open(log_path) as f:
    lines = f.readlines()
    count = 0
    for i in range(len(lines)):
        line = lines[i]
        if line[:3] == "---":
                count += 1
                # if count == 1:
                #     continue
                if (count-2)%3 == 0:
                    # get index
                    index = func_get_index(line)-1; 
                    # print(index)
                    
                    # find the last line with loss and val_loss
                    line_loss = None
                    n = 1
                    while not (lines[i-n][:2].isnumeric()):
                        n += 1    
                    line_loss = lines[i-n]

                    # get loss and val_loss
                    loss, val_loss = func_get_loss(line_loss)
                    

                    # write into csv file
                    with open("log/"+nm_tem+"log.csv", 'a') as fl:
                        write = csv.writer(fl)    
                        write.writerow([index, loss, val_loss])




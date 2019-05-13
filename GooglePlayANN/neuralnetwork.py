with open("googleplaystore.csv", "r") as f:
    lines = f.readlines()
with open("googleplaystore.csv", "w") as f:
    for line in lines:
        if "$" in line:
            line = line.replace("$","")
        f.write(line)

#%%
            
import pandas as pd

d = pd.read_csv("googleplaystore.csv")




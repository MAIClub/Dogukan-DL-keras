import numpy as np
import pandas as pd

dataset = pd.read_csv("Iris.csv")
x = dataset.iloc[:,1:4].values
y = dataset.iloc[:,-1].values



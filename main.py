import numpy as np
import pandas as pd
import sys

from sklearn.feature_selection import mutual_info_classif

dataset = sys.argv[1]
df = pd.read_csv(dataset)


## Stage 1 : Feature Selection (May contain discretization methods) 
x = df.drop(columns=[c for c in df.columns if c[-1] in ["-","+"]])
y = df[[c for c in df.columns if c[-1] in ["-","+"]]]


# Calculate Information Gain using mutual_info_classif
info_gain = mutual_info_classif(x, y)
print("Information Gain for each feature:", info_gain)

## Stage 2 : Case Selection (Active Leaarning)

## Stage 3 : Select Analogy

## Stage 4 : Make Prediction
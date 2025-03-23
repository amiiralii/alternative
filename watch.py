import csv
import sys
import os
import pandas as pd
import main
import stats
from sklearn.manifold import SpectralEmbedding
import numpy as np
import time

directory = sys.argv[1]
df = pd.read_csv(directory)
df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])
df.drop_duplicates(inplace=True)

cols = [c for c in df.columns if c[-1] not in ["+","-"]]
targets = [c for c in df.columns if c[-1] in ["+","-"]]

# Apply manifold learning to transform features
with open('intrinsic_dimension.csv', 'r', newline='') as csvfile:
    for row in csv.DictReader(csvfile):
        if row["Dataset_Name"] == directory.split("/")[-1][:-4]:
            dim = int(row["intrinsic_dimension"])
manifold_features = pd.DataFrame(SpectralEmbedding(n_components=dim).fit_transform(df[cols]))
manifold_features.columns = ["C"+str(t) for t in manifold_features.columns]

for t in targets:
    manifold_features[t] = df[t]

ss = {"FS":  "00", 
          "CS":   "2", 
          "AS":   "2", 
          "Dist": "0"}
r0 = main.experiment(sys.argv[1], ss, 5, manifold_features) 
print(111)
ss["FS"] = "11"
r1 = main.experiment(sys.argv[1], ss, 5)
r2 = main.run_baseline(sys.argv[1], "lgbm")


idx = 0
for i,j,k in zip(r0, r1, r2):
    s0, s1, s2 = stats.SOME(txt="C1"), stats.SOME(txt="simple"), stats.SOME(txt="SOTA") 
    s0.adds(i)
    s1.adds(j)
    s2.adds(k)
    print(f"Target {targets[idx]}")
    idx += 1
    stats.report([s0, s1, s2])
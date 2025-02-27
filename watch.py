import csv
import sys
import os
import pandas as pd
import main
from sklearn.preprocessing import LabelEncoder
import time

directory = sys.argv[1]
df = pd.read_csv(directory)
df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])
df.drop_duplicates(inplace=True)
le = LabelEncoder()
for c in df.columns:
    if not c[0].isupper():
        df[c] = le.fit_transform(df[c])
a = df.iloc[200]
k = df.sample(frac=0.8, ignore_index=True)
features = [c for c in df.columns if c[-1] not in ["-","+"]]
targets = [c for c in df.columns if c[-1] in ["-","+"]]
info = {}
info['head'] = list(k[features].columns)
info['min'] = list(k[features].min())
info['max'] = list(k[features].max())
info['scores'] = [1 for _ in features]   
t1 = time.time()
print(int(len(df) * 0.2))
for ii in range(int(len(df) * 0.02)):
    main.knn(k[features+[targets[0]]].values.tolist(), a[features+[targets[0]]].values.tolist(), 5, info)
    if ii==0: print("1 time:",round(time.time()-t1,2))
print("0.1 total time:",round(time.time()-t1,2))
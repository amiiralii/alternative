import csv
import sys
import os
import pandas as pd

directory = os.getcwd()
for dir in os.listdir(f"{directory}/data/optimize/"):
    for dataset in os.listdir(f"{directory}/data/optimize/{dir}"):
        if dataset[-4:]=='.csv':
            df = pd.read_csv(f"{directory}/data/optimize/{dir}/{dataset}")
            df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])
            kk = df.drop(columns=[c for c in df.columns if c[-1] in ["-","+"]])
            if sum(kk.duplicated())>0: print(f"{dataset}:",sum(kk.duplicated()))
            
import os
import pandas as pd

def save_stat(stat, cnt, filename, directory):
    if len(stat)<1:
        return
    dir = directory + "/results/parsed/" + str(cnt) + "_" + filename
    with open(dir, "w") as f:
        for s in range(len(stat)):
            f.write(stat[s])
            f.write('\n')


directory = os.getcwd()
for dirname in os.listdir(f"{directory}/data/optimize/"):
    for filename in os.listdir(f"{directory}/data/optimize/{dirname}"):
        print(f"{directory}/data/optimize/{dirname}/{filename}")
        if ".csv" in filename:
            df = pd.read_csv(f"{directory}/data/optimize/{dirname}/{filename}")
            targets = [c for c in df.columns if c[-1] in ["-","+"]]
            features = [c for c in df.columns if c[-1] not in ["-","+"]]
            for trg in targets:
                df2 = df[features+[trg]]
                df2.to_csv(f"{directory}/data/optimize2/{dirname}/{filename[:-4]}_{trg}.csv", index=False)
print("Done!")
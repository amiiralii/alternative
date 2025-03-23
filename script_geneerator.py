import os

def printscript(dataset):
  print(f'python3.12 main.py {dataset} | tee results/{dataset.split("/")[-1]}')
  
for i in os.listdir(f"data/optimize2"):
  for j in os.listdir(f"data/optimize2/{i}/"):
    if j[-4:] == ".csv":
        printscript(str("data/optimize2/"+i+"/"+j.replace(" ","")))
  
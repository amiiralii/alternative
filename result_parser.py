import os

def save_stat(stat, cnt, filename, directory):
    if len(stat)<1:
        return
    dir = directory + "/results/parsed/" + str(cnt) + "_" + filename
    with open(dir, "w") as f:
        for s in range(len(stat)):
            f.write(stat[s])
            f.write('\n')




directory = os.getcwd()
for filename in os.listdir(f"{directory}/results/"):
    if ".csv" in filename:
        with open(f"{directory}/results/{filename}", "r") as file:
            content = file.read().split("\n")
            stat = []
            tcnt = 0
            print(filename)
            for i in range(len(content)-1):
                if 'Target' in content[i] and len(stat) > 0:
                    #print("Target=",tcnt)
                    #[print(s) for s in stat]
                    save_stat(stat,tcnt,filename,directory)
                    tcnt += 1
                    stat = []
                elif i==len(content)-2:
                    stat.append(content[i])
                    #print("Target=",tcnt)
                    #[print(s) for s in stat]
                    save_stat(stat,tcnt,filename,directory)
                    tcnt += 1
                    stat = []
                else:
                    if 'Target' not in content[i]: stat.append(content[i])
print("Done!")
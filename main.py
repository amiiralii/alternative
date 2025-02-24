import numpy as np
import pandas as pd
import sys
import math
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
from ITMO_FS.filters.univariate import reliefF_measure
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import regressors
import random
from hyperopt import hp, fmin, tpe, Trials
import stats


# Gets x, max and min of the col and returns the normalized value
def norm(x,hi,lo):
    return ((x - lo) / (hi - lo + 1E-32))

## Distance Functions:
# Euclidean Distance
def dist1(r1, r2, info, weight = False):
    dst = 0
    for i in range(len(r1)-1):
        if info['head'][i][0].isupper():
            if weight:
                dst += ( info['scores'][i] * ( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) ) ** 2)
            else:
                dst += ( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) ) ** 2
        else:
            if weight:
                dst += info['scores'][i] * (r1[i]!=r2[i])
            else:
                dst += (r1[i]!=r2[i])
    if weight:
        return ( dst / ((len(r1)-1)*sum(info['scores'])) ) ** 0.5
    else:           
        return ( dst / (len(r1)-1) ) ** 0.5

# Manhattan Distance
def dist2(r1, r2, info, weight = False):
    dst = 0
    for i in range(len(r1)-1):
        if info['head'][i][0].isupper():
            if weight:
                dst += info['scores'][i] * abs( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) )
            else:
                dst += abs( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) )
        else:
            if weight:
                dst += info['scores'][i] * (r1[i]!=r2[i])
            else:
                dst += (r1[i]!=r2[i])
    if weight:
        return dst / sum(info['scores'])
    else:           
        return dst

# Pick a random candidate from todo 20 times
# Max( Min( dist( candidate , labeled ) ) )
def diversity_sampling(labeled, unlabeled, info):
    dist_picks = {}
    for _ in range(20):
        pick = random.randrange(len(unlabeled))
        dist_picks[pick] = min( dist1(l, unlabeled[pick], info, True) for l in labeled)
    return max(dist_picks, key= lambda x: dist_picks[x])

## Find K-Nearest Neighbors
def knn(labeled, row, k, info, weight = False):
    distances = []
    for l in labeled:
        distances.append([l, dist1(l,row, info, weight)])
    s = sorted(distances, key = lambda d:d[1])[:k]
    
    return [r[0] for r in s]

## Computes Weighted average based on distance between rows
def weighted_avg(cluster, target_row, info):
    distance = []
    [distance.append(dist1(c, target_row,info)) for c in cluster]

    avg, weights_sum = 0, 0 
    for i, p in enumerate(cluster):
        w = (1/(distance[i]**2 + 1E-32))
        weights_sum += w
        avg += (w) * p[-1]
 
    return avg/weights_sum

## MAPE metric for accuracy
def mape(true, pred):
    return np.mean(np.abs((true - pred) / (true + 1E-32))) * 100

## Cross Validation since our methods are not deterministic
def cross_val(dataset, repeat=5, folds=5):
    for _ in range(repeat):
        dataset = shuffle(dataset, random_state=42)
        kf = KFold(n_splits=folds, shuffle=False)
        for train_index, test_index in kf.split(dataset):
            yield dataset.iloc[train_index].reset_index(drop=True), dataset.iloc[test_index].reset_index(drop=True)        


def experiment(dataset, settings, rpt=5):
    df = pd.read_csv(dataset)
    df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])
    df.drop_duplicates(inplace=True)

    ## Set the experiment settings
    # Values for FS: 00(No feature selection), 
    #                11(info. gain with uniform discr.), 12(info. gain with quantile discr.), 13(info. gain with kmeans discr.)
    #                20(ReliefF)
    # Values for CS: 0(No case selection), 1(Random Sampling), 2(diversity sampling)
    # Values for AS: 0(No analogy/using all samples), 1(random analogies), 2(KNN)
    # Values for Dist: 0(Manhattan Distance), 1(Euclidean Distance)
    results = []
    targets = [c for c in df.columns if c[-1] in ["-","+"]]
    features = [c for c in df.columns if c[-1] not in ["-","+"]]
    for train,test in cross_val(df, repeat=rpt):    ## 5-Fold Cross Validation with 5 repeats
        ###########
        ## Stage 1 : Feature Selection (May contain discretization methods)
        ###########

        ### Discretization
        if sum(1 for col in train.columns if col[0].isupper()) == 0: settings["FS"][1] = 0
        if settings["FS"][1] == "0":
            x_train  = train.drop(columns=targets)
        else:
            if settings["FS"][1] == "1": discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')  ## Equal Width
            if settings["FS"][1] == "2": discretizer = KBinsDiscretizer(encode='ordinal', strategy='quantile') ## Equal Frequency
            if settings["FS"][1] == "3": discretizer = KBinsDiscretizer(encode='ordinal', strategy='kmeans')   ## Kmeans
            num = []
            x_train = pd.DataFrame()
            for c in train.columns: 
                if c[0].isupper() and train[c].nunique() > 10: num += [c]
                else: x_train.loc[:, c] = train.loc[:, c]
            x_train = pd.concat([x_train, pd.DataFrame(discretizer.fit_transform(train[num]), columns=num)],axis=1)
            x_train= x_train.drop(columns=targets)
        
        ### Feature Selection
        target_results = []
        for t in targets:
            y = train[t]
            if settings["FS"][0] == "0":  feature_score = [1 for _ in features]              ## No Feature Selection
            if settings["FS"][0] == "1":  feature_score = mutual_info_regression(x_train, y.values)    ## Information Gain
            if settings["FS"][0] == "2":  
                sampled_indices = x_train.sample(n=int(len(x_train)**0.5)).index
                feature_score = reliefF_measure(x_train.iloc[sampled_indices].values, y.iloc[sampled_indices].values) ## ReliefF

            for i in range(len(feature_score)): 
                if (feature_score[i] < 0 or math.isnan(feature_score[i]) ): feature_score[i] = 0 
            final_features = []
            mean_score, std_dev = np.mean(feature_score), np.std(feature_score)
            threshold = mean_score - 0.7 * std_dev
            final_features = [features[i] for i, score in enumerate(feature_score) if score >= threshold]
        
        ###########
        ## Stage 2 : Case Selection (Active Leaarning)
        ###########
            unlabeled = train[final_features+[t]].values.tolist() ## Storing training data in a list format including all usefull features and current target
            info = {}
            info['head'] = list(train[final_features].columns)
            info['min'] = list(train[final_features].min())
            info['max'] = list(train[final_features].max())
            info['scores'] = feature_score
            random.shuffle(unlabeled)
            labeled = [unlabeled.pop() for _ in range(4)] ## Warm Start with 4 Random points
            budget = math.sqrt(len(unlabeled))      ## Number of total labeled points
            if settings["CS"] == "0":   labeled = unlabeled                                                             ## No Sampling
            else:
                while (len(labeled) < budget):
                    if settings["CS"] == "1":   labeled += [unlabeled.pop(random.randint(0, len(unlabeled)-1))]             ## Random Sampling
                    if settings["CS"] == "2":   labeled += [unlabeled.pop(diversity_sampling(labeled, unlabeled, info))]   ## Diversity Sampling
                    
        ###########
        ## Stage 3 : Select Analogy
        ###########
            preds = []
            for ii,tst in test.iterrows():
                if settings["AS"] == "0":   pred_rows = labeled                                            ## No Analogy
                if settings["AS"] == "1":   pred_rows = [random.choice(labeled) for _ in range(12)]        ## Random
                if settings["AS"] == "2":   pred_rows = knn(labeled, tst[final_features+[t]].values.tolist(), 5, info)         ## KNN
        
        ###########
        ## Stage 4 : Make Prediction
        ###########
                preds.append( weighted_avg(pred_rows, tst[final_features+[t]].values.tolist(), info) )     ## Predicting using weighted average of cluster

        ###########
        ## Stage 5 : Evaluation
        ###########
            target_results.append(mape(test[t].to_numpy(),np.array(preds)))

        results.append(target_results)

    if rpt==1:
        return sum(np.median(k) for k in np.array(results).T)
    else:
        return np.array(results).T
    
    ## For train/test :
    ##      For t in target:
    ##          feature selection
    ##          sampling
    ##          For tst in test:
    ##              analogy
    ##              prediction
    ##          mape evals
    ## stats report


def run_baseline(dataset, model):
    df = pd.read_csv(dataset)
    df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])
    df.drop_duplicates(inplace=True)

    targets = [c for c in df.columns if c[-1] in ["-","+"]]
    features = [c for c in df.columns if c[-1] not in ["-","+"]]

    res = []
    ## Baselines and SOTA
    for train,test in cross_val(df):
        r=[]
        for t in targets:
            if model == "linear":   r.append(mape(test[t].to_numpy(), regressors.linear(train[features],train[t],test[features])))
            if model == "lgbm": r.append(mape(test[t].to_numpy(), regressors.lightgbm(train[features],train[t],test[features])))
        res.append(r)
    return np.array(res).T

def optimization(space):
    settings = {"FS":   space['feature_selection'], 
                "CS":   space['case_selection'], 
                "AS":   space['analogy_selection'], 
                "Dist": space['distance_function']}
    
    return experiment(sys.argv[1], settings, 1)


if __name__ == '__main__':

    space = {
        'feature_selection': hp.choice('FS', ["00", "11", "12", "13", "20"]),
        'case_selection': hp.choice('CS', ['0', '1', '2']),
        'analogy_selection': hp.choice('AS', ['0', '1', '2']),
        'distance_function': hp.choice('Dist', ['0', '1'])
    }
    best = fmin(
        fn=optimization,          # Objective function
        space=space,              # Search space
        algo=tpe.suggest,         # Tree-structured Parzen Estimator (TPE)
        max_evals=50,             # Number of trials
        trials=Trials(),           # Store results
        show_progressbar=False
    )

    best_settings = {"FS":  ["00", "11", "12", "13", "20"][best['FS']], 
                    "CS":   ['0', '1', '2'][best['CS']], 
                    "AS":   ['0', '1', '2'][best['AS']], 
                    "Dist": ['0', '1'][best['Dist']]}
    baseline_setting = {"FS":  "00", 
                    "CS":   "0", 
                    "AS":   "1", 
                    "Dist": "0"}
    
    r0, r1, r2, r3 = experiment(sys.argv[1], baseline_setting, 5), experiment(sys.argv[1], best_settings, 5), run_baseline(sys.argv[1], "linear"), run_baseline(sys.argv[1], "lgbm")

    idx = 0 
    for i,j,k,w in zip(r0, r1,r2,r3):
        s0, s1, s2, s3 = stats.SOME(txt="baseline"), stats.SOME(txt="ours"), stats.SOME(txt="lr"), stats.SOME(txt="lgbm")
        s0.adds(i)
        s1.adds(j)
        s2.adds(k)
        s3.adds(w)
        print(f"Target {idx}")
        idx += 1
        stats.report([s0, s1, s2, s3])

    #experiment(sys.argv[1], settings)
    #run_baseline(sys.argv[1])

    #for target in [c for c in df.columns if c[-1] in ["+","-"]]:
    #    print(f"\n{target}:")
    #    stats.report( [vals for st,vals in statistics.items() if target in st] )


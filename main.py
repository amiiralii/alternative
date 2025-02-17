import numpy as np
import pandas as pd
import sys
import math
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from ITMO_FS.filters.univariate import reliefF_measure
from sklearn.model_selection import KFold

import random

# Gets x, max and min of the col and returns the normalized value
def norm(x,hi,lo):
    return ((x - lo) / (hi - lo + 1E-32))

## Distance Functions:
# Euclidean Distance
def dist1(r1, r2, info):
    dst = 0
    for i in range(len(r1)-1):
        if info['head'][i][0].isupper():
            dst += abs( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) ) ** 2
        else:
            dst += (r1[i]!=r2[i])
    return ( dst / (len(r1)-1) ) ** 0.5

# Manhattan Distance
def dist2(r1, r2, info):
    dst = 0
    for i in range(len(r1)-1):
        if info['head'][i][0].isupper():
            dst += abs( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) )
        else:
            dst += (r1[i]!=r2[i])
    return dst

# Pick a random candidate from todo 20 times
# Max( Min( dist( candidate , labeled ) ) )
def diversity_sampling(labeled, unlabeled, info):
    dist_picks = {}
    for _ in range(20):
        pick = random.randrange(len(unlabeled))
        dist_picks[pick] = min( dist1(l, unlabeled[pick], info) for l in labeled)
    return max(dist_picks, key= lambda x: dist_picks[x])

## Find K-Nearest Neighbors
def knn(labeled, row, k, info):
    distances = []
    for l in labeled:
        distances.append([l, dist1(l,row, info)])
    s = sorted(distances, key = lambda d:d[1])[:k]
    return [r[0] for r in s], [r[1] for r in s]


def weighted_avg(tst, cluster, weights, info, target):
    print(tst)
    print(cluster)
    print(info['head'])
    print(target)
    input()
    v = 0
    for i in range(len(info['head'])):
        if info['head'][i][0].isupper():
            pass

    print(len(cluster.iloc[0]), len(weights), len(info['scores']))
    input()
    return 


def cross_val(dataset, repeat=5, folds=5):
    for _ in range(repeat):
        dataset = dataset.sample(frac = 1)
        kf = KFold(n_splits=folds, shuffle=False)
        for train_index, test_index in kf.split(dataset):
            yield dataset.iloc[train_index].reset_index(drop=True), dataset.iloc[test_index].reset_index(drop=True)        


df = pd.read_csv(sys.argv[1])
df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])

mln=0
for train,test in cross_val(df):    ## 5-Fold Cross Validation with 5 repeats
    print(mln)
    mln+=1
    ###########
    ## Stage 1 : Feature Selection (May contain discretization methods)
    ###########

    ### Discretization
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')  ## Equal Width
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='quantile') ## Equal Frequency
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='kmeans')   ## Kmeans
    num = []
    train_binned = pd.DataFrame()
    for c in train.columns: 
        if c[0].isupper() and train[c].nunique() > 10: num += [c]
        else: train_binned.loc[:, c] = train.loc[:, c]
    train_binned = pd.concat([train_binned, pd.DataFrame(discretizer.fit_transform(train[num]), columns=num)],axis=1)
    ### Feature Selection
    targets = [c for c in train.columns if c[-1] in ["-","+"]]
    features = [c for c in train.columns if c[-1] not in ["-","+"]]
    x_binned= train_binned.drop(columns=targets)
    x_cont  = train.drop(columns=targets)
    for t in targets:
        y = train_binned[t].values
        #feature_score = mutual_info_classif(x_binned, y)    ## Information Gain
        feature_score = reliefF_measure(x_cont.values, y)   ## ReliefF

        for i in range(len(feature_score)): 
            if (feature_score[i] < 0 or math.isnan(feature_score[i]) ): feature_score[i] = 0 
        score_sum = sum(feature_score)
        final_features = []
        for i in range(len(feature_score)):
            if feature_score[i]/score_sum > 0.07: final_features.append(features[i])
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
        labeled = [unlabeled.pop() for _ in range(4)]
        budget = math.sqrt(len(unlabeled))
        while (len(labeled) < budget):
            labeled += [unlabeled.pop(random.randint(0, len(unlabeled)-1))]             ## Random Sampling
            #labeled += [unlabeled.pop(diversity_sampling(labeled, unlabeled, info))]   ## Diversity Sampling

    ###########
    ## Stage 3 : Select Analogy
    ###########
        
        preds = []
        for _,tst in test.iterrows(): 
            #pred_rows = [random.choice(labeled) for _ in range(12)]        ## Random
            #pred_rows = labeled                                            ## No Analogy
            pred_rows, weights = knn(labeled, tst.values.tolist(), 5, info) ## KNN
    ###########
    ## Stage 4 : Make Prediction
    ###########
            cluster = pd.DataFrame(pred_rows, columns=[info['head']+[t]])
            #preds.append(float(cluster[t].mean().iloc[0]))             ## Predicting using mean of cluster
            preds.append( weighted_avg(tst, cluster, weights, info, t) )    ## Predicting using weighted average of cluster
        test[f'{t}Preds'] = preds
        print(test)
        input()


        
#           print(weighted_avg(k,row,info))

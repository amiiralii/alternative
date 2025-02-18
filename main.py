import numpy as np
import pandas as pd
import sys
import math
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from ITMO_FS.filters.univariate import reliefF_measure
from sklearn.model_selection import KFold
import regressors
import random
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
        dist_picks[pick] = min( dist1(l, unlabeled[pick], info, True) for l in labeled)
    return max(dist_picks, key= lambda x: dist_picks[x])

## Find K-Nearest Neighbors
def knn(labeled, row, k, info):
    distances = []
    for l in labeled:
        distances.append([l, dist1(l,row, info, True)])
    s = sorted(distances, key = lambda d:d[1])[:k]
    
    return [r[0] for r in s], [r[1] for r in s]

## Computes Weighted average based on distance between rows
def weighted_avg(cluster, distance, target):
    avg = 0 
    weights_sum = 0
 
    for i,tar in cluster[target].iterrows(): 
      w = (1/(distance[i]**2 + 1E-32)) 
      weights_sum += w
      avg += (w) * tar.values[0] 
    
    return avg/weights_sum


## MAPE metric for accuracy
def mape(df, true, pred):
    actual = df[true].values
    predicted = df[pred].values
    return np.mean(np.abs((actual - predicted) / (actual + 1E-32))) * 100



def cross_val(dataset, repeat=5, folds=5):
    for _ in range(repeat):
        dataset = dataset.sample(frac = 1)
        kf = KFold(n_splits=folds, shuffle=False)
        for train_index, test_index in kf.split(dataset):
            yield dataset.iloc[train_index].reset_index(drop=True), dataset.iloc[test_index].reset_index(drop=True)        


df = pd.read_csv(sys.argv[1])
df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])
df.drop_duplicates(inplace=True)

mln=0
statistics = {}
for target in [c for c in df.columns if c[-1] in ["+","-"]]:
    for method in ["knn", "linear", "lgbm", "knn-fs", "linear-fs", "lgbm-fs"]:
        statistics[f"{target}{method}"] = stats.SOME(txt=f"{target}{method}")

for train,test in cross_val(df):    ## 5-Fold Cross Validation with 5 repeats
    #print(mln)
    #mln+=1
    ###########
    ## Stage 1 : Feature Selection (May contain discretization methods)
    ###########

    ### Discretization
    #discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')  ## Equal Width
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='quantile') ## Equal Frequency
    #discretizer = KBinsDiscretizer(encode='ordinal', strategy='kmeans')   ## Kmeans
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
        feature_score2 = [1 for _ in features]              ## No Feature Selection
        #feature_score = mutual_info_classif(x_binned, y)    ## Information Gain
        feature_score = reliefF_measure(x_cont.values, y)   ## ReliefF

        for i in range(len(feature_score)): 
            if (feature_score[i] < 0 or math.isnan(feature_score[i]) ): feature_score[i] = 0 
        final_features = []
        mean_score, std_dev = np.mean(feature_score), np.std(feature_score)
        threshold = mean_score - 0.7 * std_dev
        final_features = [features[i] for i, score in enumerate(feature_score) if score >= threshold]
        final_features2 = features
        
        feature_selection_diff = False
        if final_features != final_features2: feature_selection_diff = True

    
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
            #labeled += [unlabeled.pop(random.randint(0, len(unlabeled)-1))]             ## Random Sampling
            labeled += [unlabeled.pop(diversity_sampling(labeled, unlabeled, info))]   ## Diversity Sampling
        #labeled = unlabeled                                                             ## No Sampling
        
        if feature_selection_diff:
            unlabeled2 = train[final_features2+[t]].values.tolist() ## Storing training data in a list format including all usefull features and current target
            info2 = {}
            info2['head'] = list(train[final_features2].columns)
            info2['min'] = list(train[final_features2].min())
            info2['max'] = list(train[final_features2].max())
            info2['scores'] = feature_score2
            random.shuffle(unlabeled2)
            labeled2 = [unlabeled2.pop() for _ in range(4)]
            budget = math.sqrt(len(unlabeled2))
            while (len(labeled2) < budget):
                #labeled += [unlabeled.pop(random.randint(0, len(unlabeled)-1))]             ## Random Sampling
                labeled2 += [unlabeled2.pop(diversity_sampling(labeled2, unlabeled2, info2))]   ## Diversity Sampling
            #labeled = unlabeled


    ###########
    ## Stage 3 : Select Analogy
    ###########
        
        preds, preds2 = [], []
        for _,tst in test.iterrows(): 
            #pred_rows = [random.choice(labeled) for _ in range(12)]        ## Random
            #pred_rows = labeled                                            ## No Analogy
            pred_rows, distances = knn(labeled, tst.values.tolist(), 5, info) ## KNN
            if feature_selection_diff:  pred_rows2, distances2 = knn(labeled2, tst.values.tolist(), 5, info2) ## KNN
    
    ###########
    ## Stage 4 : Make Prediction
    ###########
            cluster = pd.DataFrame(pred_rows, columns=[info['head']+[t]])
            #preds.append(float(cluster[t].mean().iloc[0]))           ## Predicting using mean of cluster
            preds.append( weighted_avg(cluster, distances, t) )     ## Predicting using weighted average of cluster
            
            if feature_selection_diff:
                cluster2 = pd.DataFrame(pred_rows2, columns=[info2['head']+[t]])
                preds2.append( weighted_avg(cluster2, distances2, t) )   ## Predicting using weighted average of cluster


    ###########
    ## Stage 5 : Evaluation
    ###########   
        test[f'{t}knn'] = preds
        test[f'{t}linear'] = regressors.linear(train[final_features],train[t],test[final_features])
        test[f'{t}lgbm'] = regressors.lightgbm(train[final_features],train[t],test[final_features])
        if feature_selection_diff:
            test[f'{t}knn-fs'] = preds2
            test[f'{t}linear-fs'] = regressors.linear(train[final_features2],train[t],test[final_features2])
            test[f'{t}lgbm-fs'] = regressors.lightgbm(train[final_features2],train[t],test[final_features2])
        else:
            test[f'{t}knn-fs'] = test[f'{t}knn']
            test[f'{t}linear-fs'] = test[f'{t}linear']
            test[f'{t}lgbm-fs'] = test[f'{t}lgbm']
        
        for method in ["knn", "linear", "lgbm", "knn-fs", "linear-fs", "lgbm-fs"]:
            statistics[f"{t}{method}"].add(mape(test,t,f'{t}{method}'))

for target in [c for c in df.columns if c[-1] in ["+","-"]]:
    print(f"\n{target}:")
    stats.report( [vals for st,vals in statistics.items() if target in st] )

## For train/test :
##      For t in target:
##          feature selection
##          sampling
##          For tst in test:
##              analogy
##              prediction
##          mape evals
## stats report
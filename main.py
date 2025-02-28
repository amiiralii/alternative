import numpy as np
import pandas as pd
import sys
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
from ITMO_FS.filters.univariate import reliefF_measure
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import regressors
import random
from hyperopt import hp, fmin, tpe, Trials
import stats
import time
from csv import DictWriter

# Gets x, max and min of the col and returns the normalized value
def norm(x,hi,lo):
    return ((x - lo) / (hi - lo + 1E-32))

# Euclidean Distance
def dist1_old(r1, r2, info, weight = False):
    dst = 0
    weights = info['scores'] if weight  else [1 for _ in range(len(info['scores']))]
    for i in range(len(r1)-1):
        if info['head'][i][0].isupper():
            dst += ( weights[i] * ( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) ) ** 2) 
        else:
            dst += weights[i] * (r1[i]!=r2[i])
    return ( dst / ((len(r1)-1)*sum(weights)) ) ** 0.5

## Vectorized version
def dist1(r1, r2, info, weight=False):
    """
    Euclidean Distance (vectorized).
    r1: either a 1D array (a single row) or a 2D array (multiple rows)
    r2: a 1D array (the query row)
    info: dictionary with keys 'head', 'min', 'max', 'scores'
    """
    # Ensure numpy arrays
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    # Determine the number of features (exclude target at end)
    n_features = r1.shape[-1] - 1
    # Use provided weights if desired; otherwise ones.
    if weight:
        weights = np.array(info['scores'], dtype=float)
    else:
        weights = np.ones(n_features, dtype=float)
    # Create a mask of numeric columns (based on column name: starts with uppercase)
    numeric_mask = np.array([col[0].isupper() for col in info['head']])

    # Extract only the feature part of the rows
    if r1.ndim == 1:
        r1_feat = r1[:n_features]
        r2_feat = r2[:n_features]
    else:
        r1_feat = r1[:, :n_features]
        r2_feat = r2[:n_features]  # r2 is assumed to be 1D
    
    # Convert info min and max to arrays
    mins = np.array(info['min'], dtype=float)
    maxs = np.array(info['max'], dtype=float)
    
    # --- Numeric part: normalize and compute squared differences ---
    if r1_feat.ndim == 1:
        r1_num = (r1_feat[numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        r2_num = (r2_feat[numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        diff_num = (r1_num - r2_num) ** 2
    else:
        r1_num = (r1_feat[:, numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        r2_num = (r2_feat[numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        diff_num = (r1_num - r2_num) ** 2

    # --- Categorical part: compute simple difference (0/1) ---
    if r1_feat.ndim == 1:
        r1_cat = r1_feat[~numeric_mask]
        r2_cat = r2_feat[~numeric_mask]
        diff_cat = (np.array(r1_cat) != np.array(r2_cat)).astype(float)
    else:
        r1_cat = r1_feat[:, ~numeric_mask]
        r2_cat = r2_feat[~numeric_mask]
        diff_cat = (r1_cat != r2_cat).astype(float)
    
    # Get weights for numeric and categorical features
    weights_num = weights[numeric_mask]
    weights_cat = weights[~numeric_mask]
    
    # Compute weighted sums (for a single row or for each row in r1)
    if r1_feat.ndim == 1:
        total = np.sum(weights_num * diff_num) + np.sum(weights_cat * diff_cat)
    else:
        total = np.sum(weights_num * diff_num, axis=1) + np.sum(weights_cat * diff_cat, axis=1)
    
    # Divisor as in the original: number of features * sum(weights)
    divisor = n_features * np.sum(weights) if weight else n_features
    
    # Return Euclidean distance
    return np.sqrt(total / divisor)

# Manhattan Distance
def dist2_old(r1, r2, info, weight = False):
    dst = 0
    weights = info['scores'] if weight  else [1 for _ in range(len(info['scores']))]
    for i in range(len(r1)-1):
        if info['head'][i][0].isupper():
            dst += weights[i] * abs( norm(r1[i], info['max'][i], info['min'][i]) - norm(r2[i], info['max'][i], info['min'][i]) )
        else:
            dst += weights[i] * (r1[i]!=r2[i])
    return dst / sum(weights)

def dist2(r1, r2, info, weight=False):
    """
    Manhattan Distance (vectorized).
    r1: either a 1D array (single row) or 2D array (multiple rows)
    r2: a 1D array (query row)
    info: dictionary with keys 'head', 'min', 'max', 'scores'
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    n_features = r1.shape[-1] - 1
    if weight:
        weights = np.array(info['scores'], dtype=float)
    else:
        weights = np.ones(n_features, dtype=float)
    numeric_mask = np.array([col[0].isupper() for col in info['head']])

    if r1.ndim == 1:
        r1_feat = r1[:n_features]
        r2_feat = r2[:n_features]
    else:
        r1_feat = r1[:, :n_features]
        r2_feat = r2[:n_features]
    
    mins = np.array(info['min'], dtype=float)
    maxs = np.array(info['max'], dtype=float)
    
    # --- Numeric part: absolute difference of normalized values ---
    if r1_feat.ndim == 1:
        r1_num = (r1_feat[numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        r2_num = (r2_feat[numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        diff_num = np.abs(r1_num - r2_num)
    else:
        r1_num = (r1_feat[:, numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        r2_num = (r2_feat[numeric_mask] - mins[numeric_mask]) / (maxs[numeric_mask] - mins[numeric_mask] + 1e-32)
        diff_num = np.abs(r1_num - r2_num)
    
    # --- Categorical part ---
    if r1_feat.ndim == 1:
        r1_cat = r1_feat[~numeric_mask]
        r2_cat = r2_feat[~numeric_mask]
        diff_cat = (np.array(r1_cat) != np.array(r2_cat)).astype(float)
    else:
        r1_cat = r1_feat[:, ~numeric_mask]
        r2_cat = r2_feat[~numeric_mask]
        diff_cat = (r1_cat != r2_cat).astype(float)
    
    weights_num = weights[numeric_mask]
    weights_cat = weights[~numeric_mask]
    
    if r1_feat.ndim == 1:
        total = np.sum(weights_num * diff_num) + np.sum(weights_cat * diff_cat)
    else:
        total = np.sum(weights_num * diff_num, axis=1) + np.sum(weights_cat * diff_cat, axis=1)
    
    divisor = np.sum(weights) if weight else 1
    return total / divisor

# Picks a random candidate from todo 20 times
# Max( Min( dist( candidate , labeled ) ) )
def diversity_sampling_old(labeled, unlabeled, info, d_func):
    dist_picks = {}
    for _ in range(20):
        pick = random.randrange(len(unlabeled))
        dist_picks[pick] = min( d_func(l, unlabeled[pick], info, True) for l in labeled)
    return max(dist_picks, key= lambda x: dist_picks[x])

## Vectorized version
def diversity_sampling(labeled, unlabeled, info, d_func):
    """
    Given a list of already labeled rows and a list of unlabeled rows,
    pick a candidate (from 20 random choices) that maximizes the minimum distance
    to the labeled set.
    """
    labeled_arr = np.array(labeled)
    unlabeled_arr = np.array(unlabeled)
    picks = np.random.choice(len(unlabeled_arr), size=min(20, len(unlabeled_arr)), replace=False)
    dist_picks = {}
    for idx in picks:
        candidate = unlabeled_arr[idx]
        dists = d_func(labeled_arr, candidate, info, True)
        dist_picks[idx] = np.min(dists)
    return max(dist_picks, key=lambda x: dist_picks[x])


## Find K-Nearest Neighbors
def knn_old(labeled, row, k, info, d_func, weight = False):
    distances = []
    t1 = time.time()
    for l in labeled:
        distances.append([l, d_func(l,row, info, weight)])
    t2 = time.time()
    s = sorted(distances, key = lambda d:d[1])[:k]
    #s = distances
    #print("sorting cost:",round(time.time() - t2,5), "iter cost:",round(t2 - t1,5), len(distances))
    
    return [r[0] for r in s]

## Vectorized version
def knn(labeled, row, k, info, d_func, weight=False):
    """
    k-Nearest Neighbors using a vectorized distance computation.
    'labeled' is a list of rows, 'row' is the query row.
    """
    labeled_arr = np.array(labeled)
    row_arr = np.array(row)
    distances = d_func(labeled_arr, row_arr, info, weight)
    idx_sorted = np.argsort(distances)[:k]
    return labeled_arr[idx_sorted].tolist()

## Computes Weighted average based on distance between rows
def weighted_avg_old(cluster, target_row, info, d_func):
    distance = []
    [distance.append(d_func(c, target_row,info)) for c in cluster]

    avg, weights_sum = 0, 0 
    for i, p in enumerate(cluster):
        w = (1/(distance[i]**2 + 1E-32))
        weights_sum += w
        avg += (w) * p[-1]
 
    return avg/weights_sum

## Vectorized version
def weighted_avg(cluster, target_row, info, d_func):
    """
    Compute the weighted average of target values from rows in the cluster.
    The weight is the inverse square of the distance between a cluster row and the target_row.
    """
    cluster_arr = np.array(cluster)
    target_arr = np.array(target_row)
    distances = d_func(cluster_arr, target_arr, info)
    weights = 1 / (distances**2 + 1e-32)
    cluster_targets = cluster_arr[:, -1]
    return np.sum(weights * cluster_targets) / np.sum(weights)

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

## Actual task happens here
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
    distance_function = dist1 if settings["Dist"]=="1" else dist2
    for train,test in cross_val(df, repeat=rpt):    ## 5-Fold Cross Validation with 5 repeats
        ###########
        ## Stage 1 : Feature Selection (May contain discretization methods)
        ###########

        ### Discretization
        le = LabelEncoder()
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
        
        for c in x_train.columns:
                if not c[0].isupper():
                    x_train[c] = le.fit_transform(x_train[c])
                    
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
            threshold = mean_score - 0.6 * std_dev
            final_features = [features[i] for i, score in enumerate(feature_score) if score >= threshold]
            final_scores = [score for i, score in enumerate(feature_score) if score >= threshold]
        ###########
        ## Stage 2 : Case Selection (Active Leaarning)
        ###########
            unlabeled = train[final_features+[t]].values.tolist() ## Storing training data in a list format including all usefull features and current target
            info = {}
            info['head'] = list(train[final_features].columns)
            info['min'] = list(train[final_features].min())
            info['max'] = list(train[final_features].max())
            info['scores'] = final_scores
            random.shuffle(unlabeled)
            labeled = [unlabeled.pop() for _ in range(4)] ## Warm Start with 4 Random points
            budget = math.sqrt(len(unlabeled))      ## Number of total labeled points
            if settings["CS"] == "0":   labeled = unlabeled                                                             ## No Sampling
            else:
                while (len(labeled) < budget):
                    if settings["CS"] == "1":   labeled += [unlabeled.pop(random.randint(0, len(unlabeled)-1))]             ## Random Sampling
                    if settings["CS"] == "2":   
                        labeled += [unlabeled.pop(diversity_sampling(labeled, unlabeled, info, distance_function))]   ## Diversity Sampling
        ###########
        ## Stage 3 : Select Analogy
        ###########
            preds = []
            for ii,tst in test.iterrows():
                if settings["AS"] == "0":   pred_rows = labeled                                            ## No Analogy
                if settings["AS"] == "1":   pred_rows = [random.choice(labeled) for _ in range(12)]        ## Random
                tst_list = tst[final_features+[t]].values.tolist()
                if settings["AS"] == "2":   pred_rows = knn(labeled, tst_list, 5, info, distance_function)         ## KNN
        ###########
        ## Stage 4 : Make Prediction
        ###########
                preds.append( weighted_avg(pred_rows, tst_list, info, distance_function) )     ## Predicting using weighted average of cluster

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

## doing same task with SOTA / common methods 
def run_baseline(dataset, model):
    df = pd.read_csv(dataset)
    df = df.drop(columns=[c for c in df.columns if c[-1] in ["X"]])
    df.drop_duplicates(inplace=True)

    targets = [c for c in df.columns if c[-1] in ["-","+"]]
    features = [c for c in df.columns if c[-1] not in ["-","+"]]
    le = LabelEncoder()
    for c in df.columns:
                if not c[0].isupper():
                    df[c] = le.fit_transform(df[c])
    res = []
    ## Baselines and SOTA
    for train,test in cross_val(df):
        r=[]
        for t in targets:
            if model == "linear":   r.append(mape(test[t].to_numpy(), regressors.linear(train[features],train[t],test[features])))
            if model == "lgbm": r.append(mape(test[t].to_numpy(), regressors.lightgbm(train[features],train[t],test[features])))
        res.append(r)
    return np.array(res).T

## function that calls experiments with different settings
def optimization(space):
    settings = {"FS":   space['feature_selection'], 
                "CS":   space['case_selection'], 
                "AS":   space['analogy_selection'], 
                "Dist": space['distance_function']}
    return experiment(sys.argv[1], settings, 1)

## Searching for a setting that optimizes our method
def find_best_setting():
    space = {
        'feature_selection': hp.choice('FS', ["00", "11", "12", "13", "20"]),
        'case_selection': hp.choice('CS', ['0', '1', '2']),
        'analogy_selection': hp.choice('AS', ['0', '1', '2']),
        'distance_function': hp.choice('Dist', ['0', '1'])
    }
    best = fmin(
        fn=optimization,
        space=space,
        algo=tpe.suggest,         # Tree-structured Parzen Estimator (TPE)
        max_evals=50,
        trials=Trials(),
        show_progressbar=False
    )

    return {"FS":  ["00", "11", "12", "13", "20"][best['FS']], 
                    "CS":   ['0', '1', '2'][best['CS']], 
                    "AS":   ['0', '1', '2'][best['AS']], 
                    "Dist": ['0', '1'][best['Dist']]}

## Translate and keeps the HPOed setting
def save_best_setting(setting):
    print(setting)
    with open('best_settings.csv', 'a') as file:
        f = DictWriter(file, fieldnames=setting.keys())
        f.writerow(setting)
        file.close()
    return


## Main Function
if __name__ == '__main__':
    ## Set the experiment settings
    # Values for FS: 00(No feature selection), 
    #                11(info. gain with uniform discr.), 12(info. gain with quantile discr.), 13(info. gain with kmeans discr.)
    #                20(ReliefF)
    # Values for CS: 0(No case selection), 1(Random Sampling), 2(diversity sampling)
    # Values for AS: 0(No analogy/using all samples), 1(random analogies), 2(KNN)
    # Values for Dist: 0(Manhattan Distance), 1(Euclidean Distance)


    #ss = {"FS":  "20", 
    #      "CS":   "2", 
    #      "AS":   "2", 
    #      "Dist": "1"}
    #t1 = time.time()
    #experiment(sys.argv[1], ss, 5)
    #input(f"done in {round(time.time()-t1,3)} seconds")


    best_settings = find_best_setting()
    baseline_setting = {"FS":  "00", 
                    "CS":   "0", 
                    "AS":   "1", 
                    "Dist": "0"}
    save_best_setting(best_settings)
    
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


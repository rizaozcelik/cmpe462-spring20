import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
#%%
def prepare_dataset():
    boston = load_boston()
    X_y = np.column_stack([boston['data'], boston['target']])
    np.random.seed(1)
    np.random.shuffle(X_y)
    X, y = X_y[:,:-1], X_y[:,-1]
    X_train, y_train, X_test, y_test = X[:400], y[:400], X[400:], y[400:]
    X_train = pd.DataFrame(X_train, columns=boston['feature_names'])
    X_test = pd.DataFrame(X_test, columns=boston['feature_names'])
    return X_train, y_train, X_test, y_test

#%%
def rss(y_left, y_right):
    def squared_residual_sum(y):
        return np.sum((y - np.mean(y)) ** 2)
    
    return squared_residual_sum(y_left) + squared_residual_sum(y_right) 
#%%
def find_best_rule(X_train, y_train):
    best_feature, best_threshold, min_rss = None, None, np.inf
    for feature in X_train.columns:
        thresholds = X_train[feature].unique().tolist()
        thresholds.sort()
        thresholds = thresholds[1:]
        for t in thresholds:
            y_left_ix = X_train[feature] < t
            y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
            t_rss = rss(y_left, y_right)
            if t_rss < min_rss:
                min_rss = t_rss
                best_threshold = t
                best_feature = feature
    
    return {'feature': best_feature, 'threshold': best_threshold}
#%%
def split(X_train, y_train, depth, max_depth):
    if depth == max_depth or len(X_train) < 2:
        return {'prediction': np.mean(y_train)}
    
    rule = find_best_rule(X_train, y_train)
    left_ix = X_train[rule['feature']] < rule['threshold']
    rule['left'] = split(X_train[left_ix], y_train[left_ix], depth + 1, max_depth)
    rule['right'] = split(X_train[~left_ix], y_train[~left_ix], depth + 1, max_depth)
    return rule
#%%
def predict(sample, rules):
    prediction = None
    while prediction is None:
        feature, threshold = rules['feature'], rules['threshold']
        if sample[feature] < threshold:
            rules = rules['left']
        else:
            rules = rules['right']
        prediction = rules.get('prediction', None)
    return prediction
#%%
def evaluate(X, y):
    preds = X.apply(predict, axis='columns', rules=rules.copy())
    return r2_score(preds, y)
#%%
X_train, y_train, X_test, y_test = prepare_dataset()
for max_depth in range(3, 11):
    rules = split(X_train, y_train, 0, max_depth)
    train_r2 = evaluate(X_train, y_train)
    test_r2 = evaluate(X_test, y_test)
    print('Max Depth', max_depth, 'R2:', train_r2, test_r2)
    
    
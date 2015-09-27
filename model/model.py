import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import pandas as pd


def cross_validate(X, y, model, window):
    '''
    Cross validates time series data using a shifting window where train data is
    always before test data
    '''
    in_sample_score = []
    out_sample_score = []
    for i in range(1, len(y)/window):
        train_index = np.arange(0, i*window)
        test_index = np.arange(i*window, (i+1)*window)
        y_train = y.take(train_index)
        y_test = y.take(test_index)
        X_train = X.take(train_index, axis=0)
        X_test = X.take(test_index, axis=0)
        model.fit(X_train, y_train)
        in_sample_score.append(model.score(X_train, y_train))
        out_sample_score.append(model.score(X_test, y_test))
        print 'Window', i
        print 'in-sample score', in_sample_score[-1]
        print 'out-sample score:', out_sample_score[-1]
        print '---'
    return model, np.mean(in_sample_score), np.mean(out_sample_score)


def fit_forest(X, y, window=100000, estimators=100,
               samples_leaf=250, validate=True):
    '''
    Fits Random Forest
    '''
    model = RandomForestRegressor(n_estimators=estimators,
                                  min_samples_leaf=samples_leaf,
                                  random_state=42,
                                  n_jobs=-1)
    if validate:
        return cross_validate(X, y, model, window)
    return model.fit(X, y)


def fit_boosting(X, y, window=100000, estimators=250, learning=.01,
                 samples_leaf=500, depth=20, validate=False):
    '''
    Fits Gradient Boosting
    '''
    model = GradientBoostingRegressor(n_estimators=estimators,
                                      learning_rate=learning,
                                      min_samples_leaf=samples_leaf,
                                      max_depth=depth,
                                      random_state=42)
    if validate:
        return cross_validate(X, y, model, window)
    return model.fit(X, y)


def grid_search(X, y, split, learn=[.01], samples_leaf=[250, 350, 500],
                depth=[10, 15]):
    '''
    Runs a grid search for GBM on split data
    '''
    for l in learn:
        for s in samples_leaf:
            for d in depth:
                model = GradientBoostingRegressor(n_estimators=250,
                                                  learning_rate=l,
                                                  min_samples_leaf=s,
                                                  max_depth=d,
                                                  random_state=42)
                model.fit(X.values[:split], y.values[:split])
                in_score = model.score(X.values[:split], y.values[:split])
                out_score = model.score(X.values[split:], y.values[split:])
                print 'learning_rate: {}, min_samples_leaf: {}, max_depth: {}'.\
                    format(l, s, d)
                print 'in-sample score:', in_score
                print 'out-sample score:', out_score
                print ''


def run_models(data, window, model_function, drop_zeros=False):
    '''
    Runs cross-validated models with a range of target offsets and outputs
    results sorted by out-of-sample performance
    '''
    mids = [col for col in data.columns if 'mid' in col]
    prevs = [col for col in data.columns if 'prev' in col]
    in_reg_scores = {}
    out_reg_scores = {}
    for i in range(len(mids)):
        print 'fitting model #{}...'.format(i+1)
        m = mids[i]
        p = prevs[i]
        if drop_zeros:
            y = data[data[m] != 0][m].values
            prev = data[data[m] != 0][p]
            X = data[data[m] != 0].drop(mids+prevs, axis=1)
            X = X.join(prev)
            X = X.values
        else:
            y = data[m].values
            prev = data[p]
            X = data.drop(mids+prevs, axis=1)
            X = X.join(prev)
            X = X.values

        _, in_reg_score, out_reg_score = model_function(X, y, window)
        in_reg_scores[m] = in_reg_score
        out_reg_scores[out_reg_score] = m

    print '\nrandom forest regressor r^2:'
    for score in sorted(out_reg_scores):
        m = out_reg_scores[score]
        print 'out-sample', m, score
        print 'in-sample', m, in_reg_scores[m], '\n'


def get_feature_importances(fitted_model, labels):
    '''
    Returns labels sorted by feature importance
    '''
    labels = np.array(labels)
    importances = fitted_model.feature_importances_
    indexes = np.argsort(importances)[::-1]
    for i in indexes:
        print '{}: {}'.format(labels[i], importances[i])
    return labels[indexes]


def get_pickle(filename):
    '''
    Pickle convenience function
    '''
    with open(filename, 'r') as f:
        data = pickle.load(f)
    return data


def append_data(df1, df2):
    '''
    Append df2 to df1
    '''
    df = pd.concat((df1, df2))
    return df.groupby(df.index).first()

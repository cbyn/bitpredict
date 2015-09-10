import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle


def cross_validate(X, y, model, window):
    '''
    Cross validates time series data using a shifting window where train
    data is always before test data
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
    return model, np.mean(in_sample_score), np.mean(out_sample_score)


def fit_regressor(X, y, window):
    '''
    Fits regressor model using cross validation
    '''
    model = RandomForestRegressor(n_estimators=50,
                                  min_samples_leaf=500,
                                  random_state=42,
                                  n_jobs=-1)
    return cross_validate(X, y, model, window)


def run_models(data, window, drop_zeros=False):
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

        _, in_reg_score, out_reg_score = fit_regressor(X, y, window)
        in_reg_scores[m] = in_reg_score
        out_reg_scores[out_reg_score] = m

    print '\nrandom forest regressor r^2:'
    for score in sorted(out_reg_scores):
        m = out_reg_scores[score]
        print 'out-of-sample', m, score
        print 'in-sample', m, in_reg_scores[m], '\n'


def print_feature_importances(fitted_model, labels):
    labels = np.array(labels)
    importances = fitted_model.feature_importances_
    indexes = np.argsort(importances)[::-1]
    for i in indexes:
        print '{}: {}'.format(labels[i], importances[i])
    return labels[indexes]


def get_pickle(filename):
    with open(filename, 'r') as f:
        data = pickle.load(f)
    return data

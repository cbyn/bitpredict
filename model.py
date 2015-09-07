import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def fit_classifier(X, y, window):
    '''
    Fits classifier model using cross validation
    '''
    y_sign = np.sign(y)
    model = RandomForestClassifier(n_estimators=100,
                                   min_samples_leaf=500,
                                   # max_depth=10,
                                   random_state=42,
                                   n_jobs=-1)
    return cross_validate(X, y_sign, model, window)


# def fit(X, y):
#     y_sign = np.sign(y)
#     model = RandomForestClassifier(n_estimators=100,
#                                    min_samples_leaf=10000,
# max_depth=10,
#                                    random_state=42,
#                                    n_jobs=-1)
#     model.fit(X[:700000], y_sign[:700000])
#     print model.score(X[:700000], y_sign[:700000])
#     print model.score(X[700000:], y_sign[700000:])
#     return model


def fit_regressor(X, y, window):
    '''
    Fits regressor model using cross validation
    '''
    model = RandomForestRegressor(n_estimators=100,
                                  min_samples_leaf=500,
                                  # max_depth=10,
                                  random_state=42,
                                  n_jobs=-1)
    return cross_validate(X, y, model, window)


def run_models(data, window):
    '''
    Runs model with a range of target offsets
    '''
    mids = [col for col in data.columns if 'mid' in col]
    in_class_scores = {}
    out_class_scores = {}
    in_reg_scores = {}
    out_reg_scores = {}
    for m in mids:
        y = data[data[m] != 0][m].values
        X = data[data[m] != 0].drop(mids).values
        _, in_class_score, out_class_score = fit_classifier(X, y, window)
        in_class_scores[m] = in_class_score
        out_class_scores[out_class_score] = m
        _, in_reg_score, out_reg_score = fit_regressor(X, y, window)
        in_reg_scores[m] = in_reg_score
        out_reg_scores[out_reg_score] = m
    print 'classifier accuracy:'
    for score in sorted(out_class_scores):
        m = out_class_scores[score]
        print 'out-of-sample', m, score
        print 'in-sample', m, in_class_scores[m]
    print 'regressor r^2:'
    for score in sorted(out_reg_scores):
        m = out_reg_scores[score]
        print 'out-of-sample', m, score
        print 'in-sample', m, in_reg_scores[m]

def get_pickle(filename):
    with open(filename, 'r') as f:
        data = pickle.load(f)
    return data

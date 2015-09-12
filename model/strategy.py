import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def run_strategy(data, cols, threshold):
    '''
    Backtests a theoretical trading strategy
    '''
    data = data[data.width > 0]
    X = data[cols]
    y = data.mid30
    split = len(y)/2
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    widths = data.iloc[split:].width/data.iloc[split:].mid
    regressor = RandomForestRegressor(n_estimators=100,
                                      min_samples_leaf=500,
                                      random_state=42,
                                      n_jobs=-1)
    regressor.fit(X_train.values, y_train.values)
    print 'r-squared', regressor.score(X_test.values, y_test.values)
    preds = regressor.predict(X_test)
    trades = np.zeros(len(preds))
    active_widths = np.zeros(len(preds))
    count = 0
    active = False
    for i, pred in enumerate(preds):
        if active:
            count += 1
            if count == 30:
                count = 0
                active = False
        elif abs(pred) > threshold:
            active = True
            trades[i] = np.sign(pred)
            active_widths[i] = (widths.iloc[i] + widths.iloc[i+30])/2

    returns = trades*y_test
    mean_return = returns[trades != 0].mean()
    print 'average return', mean_return
    profit = np.cumsum(returns)
    cum_widths = np.cumsum(active_widths)
    plt.plot(profit)
    plt.plot(cum_widths)
    plt.show()

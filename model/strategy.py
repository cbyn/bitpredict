import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.ticker as mtick


def fit_and_trade(data, cols, split, threshold):
    '''
    Fits and backtests a theoretical trading strategy
    '''
    data = data[data.width > 0]
    X = data[cols]
    y = data.mid30
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    regressor = RandomForestRegressor(n_estimators=100,
                                      min_samples_leaf=500,
                                      random_state=42,
                                      n_jobs=-1)
    regressor.fit(X_train.values, y_train.values)
    trade(X_test.values, y_test.values, regressor, threshold)


def trade(X, y, index, model, threshold):
    '''
    Backtests a theoretical trading strategy
    '''
    print 'r-squared', model.score(X, y)
    preds = model.predict(X)
    trades = np.zeros(len(preds))
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

    returns = trades*y*100
    trades_only = returns[trades != 0]
    mean_return = trades_only.mean()
    accuracy = sum(trades_only > 0)*1./len(trades_only)
    profit = np.cumsum(returns)
    plt.figure(dpi=100000)
    fig, ax = plt.subplots()
    plt.plot(index, profit)
    plt.title('Trading at Every {}% Prediction (No Transaction Costs)'
              .format(threshold*100))
    plt.ylabel('Returns')
    plt.xticks(rotation=45)
    formatter = mtick.FormatStrFormatter('%.0f%%')
    ax.yaxis.set_major_formatter(formatter)
    return_text = 'Average Return: {:.4f} %'.format(mean_return)
    trades_text = 'Total Trades: {:d}'.format(len(trades_only))
    accuracy_text = 'Accuracy: {:.2f} %'.format(accuracy*100)
    plt.text(.05, .85, return_text, transform=ax.transAxes)
    plt.text(.05, .78, trades_text, transform=ax.transAxes)
    plt.text(.05, .71, accuracy_text, transform=ax.transAxes)
    plt.show()

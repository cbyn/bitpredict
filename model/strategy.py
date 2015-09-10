import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

with open('../new_data.pkl', 'r') as f:
    data = pickle.load(f)

cols = [u'aggressor30', u'adjusted_price4', u'imbalance4', u'aggressor120',
        u'trades300', u'aggressor10', u'trend120', u'trend30', u'trades10']

data = data[data.width > 0]
X = data[cols]
y = data.mid30
X_train = X.iloc[:650000]
X_test = X.iloc[650000:]
y_train = y.iloc[:650000]
y_test = y.iloc[650000:]
widths = data.iloc[650000:].width/250
regressor = RandomForestRegressor(n_estimators=100,
                                  min_samples_leaf=500,
                                  random_state=42,
                                  n_jobs=-1)
regressor.fit(X_train.values, y_train.values)
print 'r-squared', regressor.score(X_test.values, y_test.values)
preds = regressor.predict(X_test)
trades = np.zeros(len(preds))
costs = np.zeros(len(preds))
count = 0
active = False
for i, pred in enumerate(preds):
    if active:
        count += 1
        if count == 30:
            count = 0
            active = False
    elif abs(pred) > .0005:
        active = True
        trades[i] = np.sign(pred)
        costs[i] = (widths.iloc[i] + widths.iloc[i+30])/2

returns = trades*y_test - costs
mean_return = returns[trades != 0].mean()
mean_cost = data.width.mean()/250
print 'average return', mean_return
profit = np.cumsum(returns)
plt.plot(profit)
plt.show()

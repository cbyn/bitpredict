# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

with open('data2.pkl', 'r') as f:
    data = pickle.load(f)

cols = ['width',
        'prev10',
        'adjusted_price',
        'imbalance',
        'trades30',
        'aggressor30',
        'trend30',
        'trades120',
        'aggressor120',
        'trend120']

X = data[cols]
y = data.mid10
X_train1 = X.iloc[:400000]
X_train2 = X.iloc[400000:800000]
X_test = X.iloc[800000:]
y_train1 = y.iloc[:400000]
y_train2 = y.iloc[400000:800000]
y_test = y.iloc[800000:]
regressor = RandomForestRegressor(n_estimators=50,
                                  min_samples_leaf=500,
                                  random_state=42,
                                  n_jobs=-1)

classifier = RandomForestClassifier(n_estimators=50,
                                    min_samples_leaf=500,
                                    random_state=42,
                                    n_jobs=-1)

classifier.fit(X_train1.values, np.sign(y_train1.apply(abs).values))
movement1 = classifier.predict(X_train2)
print 'accuracy', classifier.score(X_train2,
                                   np.sign(y_train2.apply(abs).values))
regressor.fit(X_train2[movement1 == 1].values, y_train2[movement1 == 1].values)
movement2 = classifier.predict(X_test)
print 'accuracy', classifier.score(X_test,
                                   np.sign(y_test.apply(abs).values))
print 'r-squared', regressor.score(X_test[movement2 == 1].values,
                                   y_test[movement2 == 1].values)
# signal = np.sign(preds)
# returns = signal*y_test
# profit = np.cumsum(returns)
# plt.plot(profit)
# plt.show()

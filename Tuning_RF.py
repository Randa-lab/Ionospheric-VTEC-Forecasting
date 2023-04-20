import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')
    
    
 max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50, None]

for count in max_depth:
    score = cross_val_score(RandomForestRegressor(max_depth = count), X_train, y_train, cv = tscv, scoring="neg_mean_squared_error")
    print(f'For max depth: {count}')
    rmse(score.mean())

    
    
max_features = [1, 2, 3, 4, 5, 6, 7]

for count in max_features:
    score = cross_val_score(RandomForestRegressor(max_features = count, max_depth = 7), X_train, y_train, cv = tscv, scoring="neg_mean_squared_error")
    print(f'For max_features: {count}')
    rmse(score.mean())

    
    
estimators = [10, 50, 100, 200, 300, 400, 500, 1000]

for count in estimators:
    score = cross_val_score(RandomForestRegressor(n_estimators = count, max_features = 3, max_depth = 7), X_train, y_train, cv = tscv, scoring="neg_mean_squared_error")
    print(f'For estimators: {count}')
    rmse(score.mean())
    

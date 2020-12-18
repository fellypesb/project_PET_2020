import pandas as pd

train = pd.read_csv('../input/best-dataset/train_average_stations2.csv',
                   usecols=['Hora','Temperatura','Umidade','Radiacao'])

test = pd.read_csv('../input/best-dataset/test_average_stations2.csv',
                  usecols=['Hora','Temperatura','Umidade','Radiacao'])

# método da exclusão

train.dropna(inplace=True)
test.dropna(inplace=True)

# seleção das variáveis preditoras e alvo

X_train = train.drop('Radiacao', axis=1)
X_test = test.drop('Radiacao', axis=1)
y_train = train['Radiacao']
y_test = test['Radiacao']

# normalização dos dados

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.1,0.9))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train.values.reshape(-1,1))
y_test = scaler.transform(y_test.values.reshape(-1,1))

# Otimização de Hiperparâmetros

import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
import numpy as np


parameters = {'hidden_layer_sizes':np.arange(2, 41, 2),
               'activation': ['logistic', 'tanh', 'relu'],
               'learning_rate': ['constant', 'adaptive'],
               'learning_rate_init': [0.0009, 0.001, 0.002, 0.004, 0.006, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8],
               'momentum': [0.0009, 0.001, 0.002, 0.004, 0.006, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9],
               'batch_size': [8,16,32,64,128,256],
               'alpha': np.power(10, np.arange(-4, 0, dtype=float))}

regressor = MLPRegressor(max_iter=200, 
                         solver='sgd',
                         hidden_layer_sizes=16,
                         activation='relu',
                         learning_rate='adaptive',
                         learning_rate_init=0.2,
                         momentum=0.8,
                         batch_size=8,
                         random_state=2020)

score = ['neg_mean_squared_error','r2']

search = RandomizedSearchCV(regressor,
                            param_distributions=parameters,
                            scoring=score,
                            n_iter=60,   # number of parameter settings that are sampled. trades off runtime vs quality of the solution.
                            n_jobs=-1,  # using all processors
                            refit='r2',
                            cv = 2,
                            random_state=2020,  # pseudo random number generator state
                            error_score=np.nan,
                            verbose=0)

start = time.time()
search.fit(X_train, y_train)
stop = time.time()

print(f'Score: {search.best_score_} \nBest_Param: {search.best_params_}\n Runtime: {stop - start}')

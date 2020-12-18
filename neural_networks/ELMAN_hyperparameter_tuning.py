import tensorflow as tf
import random as python_random
import numpy as np
import pandas as pd

train = pd.read_csv('',
                    usecols=[''])
test = pd.read_csv('',
                   usecols=[''])

# verificação de nan

train.isnull().sum()
test.isnull().sum()

# método da exclusão

train.dropna(inplace=True)
test.dropna(inplace=True)

# preparação para aplicação da janela deslizante
delay = 3
data_full = pd.concat([train, test], ignore_index=True)
test = data_full[len(data_full) - len(test) - delay:]

# Normalização dos atributos

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.1,0.9))

train_norm = scaler.fit_transform(train)
test_norm = scaler.transform(test)

# Aplicação da janela deslizante

def sliding_window(dataset, delay):
    data = np.array(dataset)
    bath = len(data)
    predictors = []
    target = []
    for i in range(delay, bath):
        predictors.append(data[i-delay:i, 0:7])
        target.append(data[i, 6])
    return (np.array(predictors), np.array(target))


X_train, y_train = sliding_window(train_norm, delay)
X_test, y_test= sliding_window(test_norm, delay)


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import SGD
from kerastuner.tuners import RandomSearch
np.random.seed(2020)
tf.random.set_seed(2020)
python_random.seed(2020)

# hypermodel

def build_model(hp):
    model = Sequential(name='Elman_RNN')

    model.add(SimpleRNN(units=hp.Int('units',
                                    min_value=2,
                                    max_value=200,
                                    step=2),
                    input_shape=(X_train.shape[1], 
                                 X_train.shape[2]),
                    activation=hp.Choice('activation',
                                         values=['relu', 'sigmoid', 'tanh']),
                    name='recurrent_layer'))

    model.add(Dense(units=1,
                    activation='linear',
                    name='output_layer'))

    model.compile(optimizer=SGD(learning_rate=hp.Choice('learning_rate',
                                                       values=list(np.logspace(-10, -0.2, base=2, num=15))),
                                momentum=hp.Choice('momentum',
                                                  values=list(np.logspace(-10, -0.1, base=2, num=15)))),
                                loss='mse', 
                                metrics=['mse'])
    return model
    
tuner = RandomSearch(
    build_model,
    objective='loss',
    max_trials=30,
    executions_per_trial=1,
    seed=2020)



tuner.search_space_summary()

tuner.search(X_train,
             y_train,
             epochs=30,
             validation_split=0.2)

tuner.get_best_hyperparameters()








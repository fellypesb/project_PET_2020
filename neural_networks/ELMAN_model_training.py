#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 23:23:35 2020

@author: fellypes barroso
"""

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


# método da exclusão de nan

train.dropna(inplace=True)
test.dropna(inplace=True)


# preparação da aplicação da janela deslizante

delay = 14
data_full = pd.concat([train, test], ignore_index=True)
test = data_full[len(data_full) - len(test) - delay:]


# escalonamento dos atributos

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.1,0.9))

train_norm = scaler.fit_transform(train)
test_norm = scaler.transform(test)


# aplicação da técnica da janela deslizante

def sliding_window(dataset, delay):
    data = np.array(dataset)
    bath = len(data)
    predictors = []
    target = []
    for i in range(delay, bath):
        predictors.append(data[i-delay:i, 0:1])
        target.append(data[i, 1])
    return (np.array(predictors), np.array(target))


X_train, y_train = sliding_window(train_norm, delay)
X_test, y_test= sliding_window(test_norm, delay)


# para modelos com apenas 1 (vetor) de entrada

X_train  = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# treinamento do modelo
np.random.seed(2020)
tf.random.set_seed(2020)
python_random.seed(2020)


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

model = Sequential(name='Elman_RNN')

model.add(SimpleRNN(units=10,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    activation='relu',
                    name='recurrent_layer'))

model.add(Dense(units=1,
                activation='linear',
                name='output_layer'))

model.compile(optimizer=SGD(learning_rate=0.125, momentum=0.93),
              loss='mse',
              metrics=['mse'])

es = EarlyStopping(monitor='loss',
                   min_delta=1e-4,
                   patience=10,
                   verbose=1)

#model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[es])



# avaliação dos modelos depois de treinados nos conjuntos de treinamento e teste
from evaluate_model import evaluate_model

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r_train = evaluate_model(y_pred_train, y_train)
r_test = evaluate_model(y_pred_test, y_test)

print('Train')
for i in r_train.Value:
    print(str(i).replace('.',','))
    
print('Test')
for i in r_test.Value:
    print(str(i).replace('.',','))



# desnormalização dos dados
x2 = scaler.inverse_transform(y_pred_test)
x1 = scaler.inverse_transform(y_test.reshape(-1,1))



import matplotlib.pyplot as plt

plt.figure(figsize=(13,5), dpi=300)
plt.title('Simple RNN', fontsize=12, weight='bold')
plt.xlabel('Tempo (h)', fontsize=12, weight='bold')
plt.ylabel('Radiação (Kj/m²)', fontsize=12, weight='bold')
plt.plot(x1,'-', label='DADOS REAIS', linewidth=2)
plt.plot(x2,'--', label='PREDIÇÕES MLP', linewidth=2)
#error = abs(x2 - x1)
# plt.plot(error, '-.', label='ERROR', linewidth=2)
plt.legend(loc='upper left', shadow=True)
plt.xlim(950, 1250)
plt.ylim(top=4000)
plt.grid(alpha=0.4)
#plt.savefig('modelo.png', dpi=300)




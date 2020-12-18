#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  15 20:13:35 2020

@author: fellype barroso
"""

import pandas as pd
import numpy as np 


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


# seleção das variáveis preditoras e alvo

X_train = train.drop('Radiacao', axis=1)
X_test = test.drop('Radiacao', axis=1)
y_train = train['Radiacao']
y_test = test['Radiacao']


# escalonamento dos atributos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.1,0.9))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train.values.reshape(-1,1))
y_test = scaler.transform(y_test.values.reshape(-1,1))

# treinamento dos modelos

from sklearn.neural_network import MLPRegressor
np.random.seed(2020)

model = MLPRegressor(solver='sgd',
                     momentum=0.8,
                     learning_rate_init=0.2,
                     learning_rate='adaptive',
                     hidden_layer_sizes=16,
                     batch_size=8,
                     alpha=0.0001,
                     activation='relu',
                    random_state=2020,
                    max_iter=200,
                    verbose=True)

model.fit(X_train, y_train)


# avaliação dos modelos
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

x1 = scaler.inverse_transform(y_test.reshape(-1,1))
x2 = scaler.inverse_transform(y_pred_test.reshape(-1,1))

# plot do gráfico de predições

import matplotlib.pyplot as plt

plt.figure(figsize=(13,5), dpi=300)
plt.title('MLP', fontsize=12, weight='bold')
plt.xlabel('Tempo (h)', fontsize=12, weight='bold')
plt.ylabel('Radiação (Kj/m²)', fontsize=12, weight='bold')
plt.plot(x1,'-', label='DADOS REAIS', linewidth=2)
plt.plot(x2,'--', label='MLP', linewidth=2)
#error = abs(x2 - x1)
#plt.plot(error, '-.', label='ERROR', linewidth=2)
plt.legend(loc='upper right', shadow=True)
plt.xlim(0, 250)
plt.ylim(top=4000)
plt.grid(alpha=0.4)
# plt.savefig('grafico_predicoes.png', dpi=300)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:55:54 2020

@author: fellypesb
"""


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-7,7,50)

def tah(x):
    return np.tanh(x)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def relu(x):
    a = []
    for i in x:
        if i > 0:
            a.append(i)
        else:
            a.append(0)
    return a
        
plt.subplot(1,2,1)
plt.plot(x, tah(x), label='$f(x)$ = tanh')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
# plt.grid()
# plt.legend()
plt.subplot(1,2,1)
plt.plot(x, sigmoid(x), label='$f(x)$ = sigmoid')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.grid()
plt.legend(loc='upper left', frameon=False, fontsize=7)
plt.subplot(1,2,2)
plt.plot(x, relu(x), color='red',label='$f(x)$ = relu')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper left', frameon=False, fontsize=7)
plt.grid()
# plt.savefig('funcoes_ativaticao.png', dpi=300)


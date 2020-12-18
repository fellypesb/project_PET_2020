#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:59:57 2020

@author: fellypesb
"""

import pandas as pd

df = pd.read_csv('/home/fellypesb/Documents/PET/project_codes/outubro/dataset/cluster_data_hour.csv',
                 parse_dates=['Data'])
df2 = df.iloc[:,:-2]


day = []
month = []
year = []

for i in df.Data:
    day.append(i.day)
    month.append(i.month)
    year.append(i.year)
    
df2['Dia'] = day
df2['Mes'] = month
df2['Ano'] = year


df2 = pd.concat([df2,df.iloc[:,-2:]], axis=1)

train = df2[:int(0.8*len(df2)) + 17]
test = df2[int(0.8*len(df2)) + 17:]

# group_train = train.groupby('Local', axis=0)
# group_test = test.groupby('Local', axis=0)

# group_train.count()
# group_test.count()

# train.to_csv('train_cdh2.csv', index=False)
# test.to_csv('test_cdh2.csv', index=False)


df = pd.read_csv('/home/fellypesb/Documents/PET/project_codes/outubro/dataset/cluster_dh_add_imputer_nan.csv',
                 parse_dates=['Data'])
df2 = df.iloc[:,:-2]


day = []
month = []
year = []

for i in df.Data:
    day.append(i.day)
    month.append(i.month)
    year.append(i.year)
    
df2['Dia'] = day
df2['Mes'] = month
df2['Ano'] = year


df2 = pd.concat([df2,df.iloc[:,-2:]], axis=1)

train = df2[:int(0.8*len(df2)) + 17]
test = df2[int(0.8*len(df2)) + 17:]

# train.to_csv('train_cdh_nan2.csv', index=False)
# test.to_csv('test_cdh_nan2.csv', index=False)


df = pd.read_csv('/home/fellypesb/Documents/PET/project_codes/outubro/dataset/average_stations.csv',
                 parse_dates=['Data'])

df2 = df.iloc[:,:-1]

day = []
month = []
year = []

for i in df.Data:
    day.append(i.day)
    month.append(i.month)
    year.append(i.year)
    
df2['Dia'] = day
df2['Mes'] = month
df2['Ano'] = year

df2 = pd.concat([df2,df.iloc[:,-1]], axis=1)

train = df2[:int(0.8*len(df2)) + 3]
test = df2[int(0.8*len(df2)) + 3:]

# train.to_csv('train_average_stations2.csv')
# test.to_csv('test_average_stations2.csv')





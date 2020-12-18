#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:48:26 2020

@author: fellypesb
"""

# Médias das variáveis meteorológica por hora. Necessário para a substituição
# dos Nan

import pandas as pd

df = pd.read_csv('/home/fellypesb/Documents/PET/project_codes/outubro/dataset/cluster_data_hour.csv')

hgroup = df.groupby(['Hora'], axis=0)

hgroup.count()

average_hourly = hgroup['Temperatura', 'Umidade', 'Radiacao'].mean()

#average_hourly.to_csv('average_hourly.csv')

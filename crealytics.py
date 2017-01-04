#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:14:13 2017

@author: sean
"""

import pandas as pd
import numpy as np

df_test=pd.read_csv('report_2016_02_01-2016_12_01.csv.gz',nrows=10000)

df_test.to_csv('report_2016_02_01-2016_12_01_10000.csv')
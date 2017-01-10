# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 14:35:57 2014

@author: sean
"""
import scipy.io as spio

import pandas as pd
import libLinear_functions

df=pd.read_csv('../data/20140623T0100_20140623T0200_ext.csv')
features="site+ad+banner_width*banner_height+site*banner_width*banner_height".split('+')
factors=features + ['site*ad']
non_factors=[]

sc, click_no_click_df, weights, targets \
        = libLinear_functions.create_sparse_cat(df, factors, non_factors)
sparse_train_all = libLinear_functions.create_sparse(sc, cut_off, click_no_click_df)
spio.mmwrite('../data/20140623T0100_20140623T0200_ext.mmx',sparse_train_all)

click_no_click=pd.DataFrame({'clicks':df.clicks, 'non_clicks':df.instances-df.clicks})
click_no_click.to_csv('../data/20140623T0100_20140623T0200_ext_click_no_click.csv', index=False)

folds=df.timeslot+1
folds.to_csv('../data/20140623T0100_20140623T0200_ext_folds.csv', index=False)

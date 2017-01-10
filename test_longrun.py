# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 13:20:16 2014

@author: sean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:39:00 2014

@author: sean
"""

import sys
import os
import numpy as np
import pandas as pd
import libLinear_functions
import crossval

data_path='../data/'
executable_path = '../../../../lib/liblinear-weights/'
#trial_factors_list=[['site'],['site','ad'],['site','ad','banner_width*banner_height'],['site','ad','banner_width*banner_height', 'device'],
#                    ['site','ad','banner_width*banner_height', 'device', 'site*banner_width*banner_height', 'site*ad' ],
#                    ['site','ad','banner_width*banner_height', 'device', 'site*banner_width*banner_height', 'site*ad', 'site*device' ]]

trial_factors_list = [['site','ad','banner_width*banner_height', 'device', 'site*banner_width*banner_height',
                       'site*ad', 'site*device' ,'hour','dayofweek','hour*campaign_id', 'dayofweek*campaign_id']]
                       
features = list(set([item for sublist in trial_factors_list[:] for item in sublist]))
# "site+ad+banner_width*banner_height+site*banner_width*banner_height"

rtb_flag = [1,0]
sql_fields = list(set(sum([fs.split('*') for fs in features], [])))
sql_table = 'aggregated_ctr' #Data table
factors = features[:]
non_factors = []

cut_off_list=[0]
C_list=[ 0.372]
#train_per_list=[['2014-06-23 01:00:00', '2014-07-07 00:00:00']]
#train_per_list=[['2014-06-23 01:00:00', '2014-07-07 00:00:00']]
train_per_start=['2014-06-23 01:00:00', '2014-07-07 00:00:00']
#train_per_start=['2014-06-23 01:00:00', '2014-06-24 00:00:00']

train_per_list=crossval.gen_dates(train_per_start,48, 1, 6, 4)

#train_per_list=train_per_list[4:]

cut_off_list=[0]
C_list=[0.372]

results_all,_ = crossval.train_loop(train_per_list,cut_off_list,C_list,factors,non_factors, data_path, executable_path, trial_factors_list)

#results_all_df=pd.DataFrame(results_all,columns=['dates','cutoff','C','logloss'])
#results_all_df.to_csv('../data/res_all1.csv')
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:39:00 2014

@author: sean
"""

import crossval

data_path='../data/'
executable_path = '../../../../lib/liblinear-weights/'

features="site+ad+banner_width*banner_height+site*banner_width*banner_height".split('+')
rtb_flag	= [1,0]
sql_fields = list(set(sum([fs.split('*') for fs in features], [])))
sql_table = 'aggregated_ctr' #Data table
factors = features[:]
non_factors = []

cut_off_list=[0]
C_list=[ 0.372]
#train_per_list=[['2014-06-23 01:00:00', '2014-07-07 00:00:00']]
#train_per_list=[['2014-06-23 01:00:00', '2014-07-07 00:00:00']]
train_per_start=['2014-06-23 01:00:00', '2014-06-24 00:00:00']

train_per_list=crossval.gen_dates(train_per_start,48, 1, 6, 4)



cut_off_list=[0]
C_list=[ 0.0372, 0.372, 3.72]


results_all = crossval.stepwise_regression(train_per_list,cut_off_list,C_list,factors,non_factors, data_path, executable_path)

#results_all_df=pd.DataFrame(results_all,columns=['dates','cutoff','C','logloss'])
#results_all_df.to_csv('../data/res_all1.csv')
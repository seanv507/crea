# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:00:56 2014

@author: sean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 17:39:00 2014

@author: sean
"""

import sys
import pandas as pd
import libLinear_functions

data_path='../data/'
executable_path = '../../../../lib/liblinear-weights/'

features="site+ad+banner_width*banner_height+site*banner_width*banner_height".split('+')

rtb_flag	= [1,0]
sql_fields = list(set(sum([fs.split('*') for fs in features], [])))
sql_table = 'aggregated_ctr' #Data table
con_dict_mad={'host':'db.lqm.io', 'db':'madvertise_production',  
              'user':'readonly', 'passwd':'z0q909TVZj'}
con_dict_dse={'host':'db.lqm.io', 'db':'dse', 'user':'dse', 'passwd':'dSe@lQm'}

factors = features[:]
non_factors = []

#train_per=['2014-06-23 01:00:00', '2014-07-07 00:00:00']

train_per=['2014-06-23 01:00:00', '2014-06-24 00:00:00']

cut_off_list=[0]
C_list=[ 0.372]
# C_list=[0.372]
train_per_list = libLinear_functions.gen_dates(train_per,48, 1, 6, 1)
test_per_list = map(lambda x: ( libLinear_functions.add_hour(x[1], 1), 
                              libLinear_functions.add_hour(x[1], 3)), 
                    train_per_list)
#def train_loop(train_per_list, cut_off_list, C_list,
#               factors,non_factors, data_path, executable_path):


# remove cross terms
factors+=['campaign_id', 'ad_account_id', 'pub_account_id', 
              'site*ad', 'ad*pub_account_id']
select_factors_list=[ features + ['site*ad']]


#select_factors_list=[features[:], 
#                    features + ['site*ad'],
#                    ['site', 'banner_width*banner_height', 'site*banner_width*banner_height', 'pub_account_id', 'ad*pub_account_id']]
rtb_flag=[0,1]
model_type=0
has_intercept = True  # bias term in LR
tol = 0.00000001
# NB these filenames are HARDCODED in write_sparse routines
weights_file = 'train_ais.txt'
train_file = 'train_svm.txt'
test_file = 'test_svm.txt'
probability_file = 'preds_SummModel_py.txt'
results = []
for train_per,test_per in zip(train_per_list, test_per_list):
    
    # DATA RANGE IS INCLUSIVE => 00:00-02:00 = 3 HOURS
#    train_df=libLinear_functions.MySQL_getdata(con_dict_dse['host'],
#                           con_dict_dse['db'],
#                           con_dict_dse['user'],
#                           con_dict_dse['passwd'],
#                           sql_table, train_per, sql_features, rtb_flag)
#    train_df=libLinear_functions.add_features(train_df)
#    test_df= libLinear_functions.MySQL_getdata(con_dict_dse['host'],
#                           con_dict_dse['db'],
#                           con_dict_dse['user'],
#                           con_dict_dse['passwd'],
#                           sql_table, test_per, sql_features, rtb_flag)
#    test_df = libLinear_functions.add_features(test_df)
#   
    train_df=libLinear_functions.load_data(data_path,train_per)
#    train_df=train_df[train_df.campaign_id==4310]
    test_df=libLinear_functions.load_data(data_path,test_per)
#    test_df=test_df[test_df.campaign_id==4310]
    sc, click_no_click_df, weights, targets \
        = libLinear_functions.create_sparse_cat(train_df, factors, non_factors)

    for cut_off in cut_off_list:
        sparse_train_all = libLinear_functions.create_sparse(sc, cut_off, click_no_click_df)
        sparse_test_all = sc.transform(test_df)
        for trial_factors in select_factors_list:        
            sparse_train=sc.select_factors(sparse_train_all, trial_factors)        
            sparse_test = sc.select_factors(sparse_test_all, trial_factors)
            
            libLinear_functions.write_sparse(sc, sparse_train, weights, targets, 
                                             data_path, len(trial_factors))
            libLinear_functions.write_sparse_test(sc, sparse_test, data_path,
                                                      n_columns_used= len(trial_factors))
    
            for C in C_list:
                mean_log_loss_train = 0
                mean_log_loss_test = 0
                model_file = \
                    '{start}_{stop}_cut_{cut_off}_C_{C:0.3}.model'.format(
                        start=libLinear_functions.date_name(train_per[0]),
                        stop=libLinear_functions.date_name(train_per[1]),
                        cut_off=cut_off, C=C)
                libLinear_functions.fit(executable_path, data_path, train_file,
                    model_file, weights_file, model_type, reg_param=C, tol=tol,
                    has_intercept=has_intercept)
                    
                pCTR_train = libLinear_functions.predict(executable_path, data_path, train_file,
                               model_file, probability_file)
                if type(pCTR_train) is pd.Series:                               
                    pCTR_train = pCTR_train.iloc[:pCTR_train.shape[0]/2]
                    # training data is duplicated 1st no -clicks then clicks
                    
                
                pCTR=pCTR_train
                data_df=train_df
                if type(pCTR) is pd.Series:
                    amounts = pd.DataFrame({
                    'no_clicks':data_df['instances' ]-data_df['clicks'],
                    'clicks':data_df['clicks']})
                    mean_log_loss, weighted_log_loss = libLinear_functions.log_loss_weighted(pCTR, amounts)
                    data_df['pCTR: ' + '+'.join(trial_factors)  ] =pCTR
                    data_df['lloss: ' + '+'.join(trial_factors)  ] =weighted_log_loss
                    mean_log_loss_train=mean_log_loss
                    
                    
                    # TODO  what to do if ERROR?
                pCTR_test = libLinear_functions.predict(executable_path, data_path, test_file,
                               model_file, probability_file)
                pCTR=pCTR_test
                data_df=test_df
                if type(pCTR) is pd.Series:
                    amounts = pd.DataFrame({
                    'no_clicks':data_df['instances' ]-data_df['clicks'],
                    'clicks':data_df['clicks']})
                    mean_log_loss, weighted_log_loss = libLinear_functions.log_loss_weighted(pCTR, amounts)
                    data_df['pCTR: ' + '+'.join(trial_factors)  ] =pCTR
                    data_df['lloss: ' + '+'.join(trial_factors)  ] =weighted_log_loss
                    mean_log_loss_test=mean_log_loss

                results.append([train_per[:], trial_factors, cut_off, C,
                                mean_log_loss_train, mean_log_loss_test])
    data_per=train_per
    data_file = data_path + 'data_pred_df_{0}_{1}4310.csv'.format(
            libLinear_functions.date_name(data_per[0]), 
            libLinear_functions.date_name(data_per[1]))

    train_df.to_csv(data_file, index=False)

    data_per=test_per
    data_file = data_path + 'data_pred_df_{0}_{1}4310.csv'.format(
            libLinear_functions.date_name(data_per[0]), 
            libLinear_functions.date_name(data_per[1]))

    test_df.to_csv(data_file, index=False)


    
#results_all_df=pd.DataFrame(results_all,columns=['dates','cutoff','C','logloss'])
#results_all_df.to_csv('../data/res_all1.csv')
"""
    Functions to iterate over different slices of training test data 
    and perform stepwise regression
"""
import os
import libLinear_functions
import gen_features
import mysql_lqm

import numpy as np
import pandas as pd


from datetime import datetime, timedelta


def subtract_hour(datetime_str, hour):
    mytime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    mytime -= timedelta(hours=hour)
    return mytime.strftime("%Y-%m-%d %H:%M:%S")


def add_hour(datetime_str, hour):
    mytime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    mytime += timedelta(hours=hour)
    return mytime.strftime("%Y-%m-%d %H:%M:%S")


def date_name(datetime_str):
    d = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    return datetime.strftime(d, '%Y%m%dT%H%M')


def gen_dates(train_per_start, hours_inc=48, n_inc=10, hours1_inc=6, n1_inc=4):
    """ generates a list of date intervals for cross validation

    nested loop because we want to try different dates
    and 3 hour prediction time window means we need to also start
    at different times of day
        """
    dates = []
    train_per = train_per_start[:]
    for i_inc in range(n_inc):
        # '2014-06-24 01:00:00','2014-06-30 00:00:00'
        train_per1 = train_per[:]
        for i1_inc in range(n1_inc):
            dates.append(train_per1[:])
            train_per1[0] = add_hour(train_per1[0], hours1_inc)
            train_per1[1] = add_hour(train_per1[1], hours1_inc)
        train_per[0] = add_hour(train_per[0], hours_inc)
        train_per[1] = add_hour(train_per[1], hours_inc)
    return dates


# is there a way to add random grouping to sql

# window
# given hour window for 2 weeks + 1 day
# aggregate over middle, and separate over the first 24 hours and last 24 hours
# then take data set - add 24 hours.

"""
retrieve data with date
group data by cross validation
create separate dataframes
pass each to one hot encoding
write to new train file
write to new test file
"""


def make_kfold(train_df, factors, n_folds, fold_col, new_fold_col='fold'):
    """ split 1 column (eg datetime) between n_folds

        some folds will have more than others
        - unless fold column is multiple of n_folds
    """
    fold_uniq = train_df[fold_col].unique()  # eg get unique dates in rows
    col_length = fold_uniq.shape[0]
    # create random ordering of fold
    n_reps = np.int(np.ceil(float(col_length)/n_folds))
    perms = np.tile(np.arange(n_folds), n_reps)
    perms1 = np.random.permutation(perms)[:col_length]
    # create series containing fold index for each row of train_df
    new_fold = train_df[fold_col].map(dict(zip(fold_uniq, perms1)))
    new_fold.name = new_fold_col
    # insert doesn't return reference
    groupby = factors[:]
    groupby.remove(fold_col)
    # using series allows us to name index column
    groupby.insert(0, new_fold)

    # groupby takes either grouping column (1 for each row) or column name
    fold_data = train_df.groupby(groupby).sum()
    # fold_data.reset_index(groupby_list,inplace=True)
    # index_names = list(fold_data.index.names[1:])
    # index_names.insert(0, new_fold_col)
    # fold_data.index.names = index_names
    return fold_data
    # needs to have indices reset to columns and remove fold index


def split_fold(fold_data, n_folds, new_fold_col='fold'):
    i_fold = 0
    indices = list(fold_data.index.names)
    # this is unnecc if fold is 1st column in index
    slicers = [slice(None)]*len(fold_data.index.names)
    fold_index = fold_data.index.names.index(new_fold_col)
    indices.remove(new_fold_col)

    while (i_fold < n_folds):
        slicers[fold_index] = [i for i in range(n_folds) if i != i_fold]
        slicers_tuple = tuple(slicers)
        train_data = fold_data.loc[slicers_tuple, :].groupby(
            level=indices).sum()
        val_data = fold_data.xs(i_fold, level=new_fold_col)
        yield train_data, val_data
        i_fold += 1


def write_kfolds(fold_data, n_folds, factors, non_factors, cut_off,
                 data_path, new_fold_col='fold',
                 hits_trials=('clicks', 'instances')):

    hits_field, trials_field = hits_trials

    for i_fold, (train_df, val_df) in enumerate(
            split_fold(fold_data, n_folds, new_fold_col)):
        train_df.reset_index(inplace=True)
        val_df.reset_index(inplace=True)
        sc, click_no_click_df, weights, targets \
            = libLinear_functions.create_sparse_cat(
                train_df, factors, non_factors, hits_trials)
        mad_sparse_train = libLinear_functions.create_sparse(
            sc, cut_off, click_no_click_df, weights)

        n_columns_used = len(sc.factors) + len(sc.non_factors)
        new_data_path = data_path + 'fold_{0}/'.format(i_fold)
        if not os.path.exists(new_data_path):
            os.makedirs(new_data_path)

        libLinear_functions.write_sparse(
            sc, mad_sparse_train, weights, targets, new_data_path)
        mad_sparse_val = sc.transform(val_df)
        ''' from libsvm readme
        The format of training and testing data file is:

        <label> <index1>:<value1> <index2>:<value2> ...
        Each line contains an instance and is ended by a '\n' character.  For
        classification, <label> is an integer indicating the class label
        (multi-class is supported). For regression, <label> is the target
        value which can be any real number. For one-class SVM, it's not used
        so can be any number.  The pair <index>:<value> gives a feature
        (attribute) value: <index> is an integer starting from 1 and <value>
        is a real number. The only exception is the precomputed kernel, where
        <index> starts from 0; see the section of precomputed kernels. Indices
        must be in ASCENDING order.
        Labels in the testing file are only used to calculate accuracy
        or errors.
        If they are unknown, just fill the first column with any numbers
        '''
        targets = np.zeros((mad_sparse_val.shape[0], ))  # just put any numbers
        gen_features.csr_write_libsvm(new_data_path + 'val_svm.txt',
                                      mad_sparse_val, targets, n_columns_used)
        non_clicks = val_df[trials_field] - val_df[hits_field]
        pd.DataFrame([non_clicks, val_df[hits_field]]).to_csv(
            new_data_path + 'val_non_clicks_clicks.txt')

# Note that the aggregated data format causes a number of issues
# training data is duplicated into no clicks and clicks
# this means that crossvalidation etc has to be 'specially' designed
# ie get indices on original data set then duplicate for clicks no clicks
# also cross val needs to be done before aggregation [eg splitting by hour?]

def load_data(data_path, data_per):
    data_file = data_path + 'data_df_{0}_{1}.csv'.format(
            date_name(data_per[0]), date_name(data_per[1]))
    data_df = pd.read_csv(data_file)
    return data_df


def MySQL_save_data_loop(con_dict, sql_table,
                         data_per_list, features, rtb_flag, data_path):

    for data_per in data_per_list:
        file_name = data_path + 'data_df_{0}_{1}.csv'.format(
            date_name(data_per[0]), date_name(data_per[1]))
        if not os.path.exists(file_name):
            data_df = mysql_lqm.MySQL_getdata(con_dict,
                                    sql_table, data_per, features, rtb_flag)
            data_df = mysql_lqm.add_features(data_df)
            data_df.to_csv(file_name, index=False)


def train_loop(train_per_list, cut_off_list, C_list,
               factors, non_factors, data_path, executable_path, 
               trial_factors_list=None):
    """ goes through data, loading from database, trying different models
    
    would be nice if had state, so can easily stop or start these multiple loops
    rather than for loops etc.
    """    
    if trial_factors_list is None:
        trial_factors_list=[factors]
    sql_table = 'aggregated_ctr' #Data table
    # remove cross terms
    sql_features = list(set(sum([fs.split('*') for fs in factors], [])))
#    factors+=['campaign_id','ad_account_id','pub_account_id', 
#                  'campaign_id*site', 'ad*pub_account_id']
    con_dict_dse={'host':'db.lqm.io','db':'dse',
                  'user':'dse','passwd':'dSe@lQm'}
    con_dict_mad={'host':'db.lqm.io','db':'madvertise_production',
                  'user':'readonly','passwd':'z0q909TVZj'}
    
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
    for train_per in train_per_list:
        test_per = ( add_hour(train_per[1], 1), add_hour(train_per[1], 3))
        # DATA RANGE IS INCLUSIVE => 00:00-02:00 = 3 HOURS
        train_df=mysql_lqm.MySQL_getdata(con_dict_dse,
                               sql_table, train_per, sql_features, rtb_flag)
        train_df=mysql_lqm.add_features( train_df)
        test_df= mysql_lqm.MySQL_getdata(con_dict_dse,
                               sql_table, test_per, sql_features, rtb_flag)
        test_df = mysql_lqm.add_features(test_df)
        
        sc, click_no_click_df, weights, targets \
            = libLinear_functions.create_sparse_cat(train_df, factors, non_factors)

    
        for cut_off in cut_off_list:
            sparse_train_all = libLinear_functions.create_sparse(sc, cut_off, click_no_click_df)
            sparse_test_all = sc.transform(test_df)
            for trial_factors in trial_factors_list:
                trial_factors=trial_factors[:] # copy
                trial_factors.sort(key=lambda x: sc.factors.index(x))
                # libsvm expects the indices in ascending order
                print (trial_factors)                    
                sparse_train=sc.select_factors(sparse_train_all, trial_factors)
                sparse_test=sc.select_factors(sparse_test_all, trial_factors)
                libLinear_functions.write_sparse(sc, sparse_train, weights, targets, data_path, len(trial_factors))
                libLinear_functions.write_sparse_test(sc, sparse_test, data_path, n_columns_used= len(trial_factors))


                for C in C_list:
                    model_file = \
                        '{start}_{stop}_cut_{cut_off}_C_{C:0.3}.model'.format(
                            start=date_name(train_per[0]),
                            stop=date_name(train_per[1]),
                            cut_off=cut_off, C=C)
                    fit(executable_path, data_path, train_file,
                        model_file, weights_file, model_type, reg_param=C, tol=tol,
                        has_intercept=has_intercept)
    
    
                    pCTR = libLinear_functions.predict(executable_path, data_path, test_file,
                                   model_file, probability_file)
                    if type(pCTR) is pd.Series:
                        amounts = pd.DataFrame({
                        'no_clicks':test_df['instances' ]-test_df['clicks'],
                        'clicks':test_df['clicks']})
                        mean_log_loss, weighted_log_loss = log_loss_weighted(pCTR, amounts)
                        results.append([train_per[:],trial_factors[:],
                        cut_off,C,amounts.clicks.sum(),amounts.no_clicks.sum(), mean_log_loss])
                        results_df=pd.DataFrame(results,columns=['date','features','cutoff','C','clicks','no_clicks','lloss'])
                        results_df.to_csv(data_path+'resultsX.txt',index=False, sep='|')
                    # what to do if ERROR?
    return results_df, weighted_log_loss
    
    
def stepwise_regression(train_per_list, cut_off_list, C_list,
               factors,non_factors, data_path, executable_path):
    """ perform stepwise input regression based on best log loss.
    
    data for each 'fold' is first read from DB and saved, 
    then the stepwise regression starts.
    The data files have timestamp names and are not deleted.
    This is because at every iteration 
    (adding a new feature) we have to go through each data set again.
    
    """
    sql_table = 'aggregated_ctr' #Data table
    sql_features = list(set(sum([fs.split('*') for fs in factors], [])))
    # remove cross terms

    factors+=['campaign_id','ad_account_id','pub_account_id', 
                  'campaign_id*site', 'ad*pub_account_id']
    con_dict_mad={'host':'db.lqm.io','db':'madvertise_production',
                  'user':'readonly','passwd':'z0q909TVZj'}
    con_dict_dse={'host':'db.lqm.io','db':'dse','user':'dse','passwd':'dSe@lQm'}
    rtb_flag=[0,1]
    
    test_per_list= map(lambda x: ( add_hour(x[1], 1), add_hour(x[1], 3)), train_per_list)
    
    # test period is next 3 hours after end of training period
    # DATA RANGE IS INCLUSIVE => 00:00-02:00 = 3 HOURS
    MySQL_save_data_loop(con_dict_dse, sql_table,
                         train_per_list, sql_features, rtb_flag, data_path)
    MySQL_save_data_loop(con_dict_dse, sql_table,
                         test_per_list, sql_features, rtb_flag, data_path)
    
    model_type=0
    has_intercept = True  # bias term in LR
    tol = 0.00000001

    # NB these filenames are HARDCODED in write_sparse routines
    weights_file = 'train_ais.txt'
    train_file = 'train_svm.txt'
    test_file = 'test_svm.txt'
    probability_file = 'preds_SummModel_py.txt'

    
    res_df_list=[]
    trial_factors=[]
    remaining_factors=factors[:]
    while len(remaining_factors):
        results = []    
        #  we assume we cannot load all the data in memory
        #  so we have to reload for every step of stepwise selection
        for train_per, test_per in zip(train_per_list, test_per_list):
            
            train_df=load_data(data_path,train_per)
            test_df=load_data(data_path,test_per)
                        
            sc, click_no_click_df, weights, targets \
                = libLinear_functions.create_sparse_cat(train_df, factors, non_factors)
    
            for cut_off in cut_off_list:
                sparse_train_all = libLinear_functions.create_sparse(sc, cut_off, click_no_click_df)
                sparse_test_all = sc.transform(test_df)
                for fac in remaining_factors:
                    trial_factors.append(fac)
                    trial_factors.sort(key=lambda x: sc.factors.index(x))
                    # libsvm expects the indices in ascending order
                    print (trial_factors)                    
                    sparse_train=sc.select_factors(sparse_train_all, trial_factors)
                    sparse_test=sc.select_factors(sparse_test_all, trial_factors)
                    libLinear_functions.write_sparse(sc, sparse_train, weights, targets, data_path, len(trial_factors))
                    libLinear_functions.write_sparse_test(sc, sparse_test, data_path, n_columns_used= len(trial_factors))

                    for C in C_list:
                        model_file = \
                            '{start}_{stop}_cut_{cut_off}_C_{C:0.3}.model'.format(
                                start=date_name(train_per[0]),
                                stop=date_name(train_per[1]),
                                cut_off=cut_off, C=C)
                        fit(executable_path, data_path, train_file,
                            model_file, weights_file, model_type, reg_param=C, tol=tol,
                            has_intercept=has_intercept)
        
                        pCTR = libLinear_functions.predict(
                            executable_path, data_path, test_file,
                                       model_file, probability_file)
                        if type(pCTR) is pd.Series:
                            amounts = pd.DataFrame({
                            'no_clicks':test_df['instances' ]-test_df['clicks'],
                            'clicks':test_df['clicks']})
                            mean_log_loss, weighted_log_loss =\
                                libLinear_functions.log_loss_weighted(pCTR, amounts)
                            results.append([train_per[:], tuple(trial_factors),fac, cut_off, C, mean_log_loss])
                    # what to do if ERROR?
                    trial_factors.remove(fac)
        res_df=pd.DataFrame(results,columns=['train_per','factors','add_factor','cut_off','C','mean_log_loss'])
        res_avg=res_df.groupby(['factors','add_factor','cut_off','C']).agg([np.mean,np.std])
        best_params=res_avg['mean_log_loss','mean'].argmin()
        best_fac=best_params[1]
        remaining_factors.remove(best_fac)
        trial_factors.append(best_fac)
        res_df_list.append(res_df)
    results_df=pd.concat(res_df_list)
    return results_df
        

        
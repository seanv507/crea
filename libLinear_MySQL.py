#from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Wed Marcs 05 14:47:41 2014

@author: tan
"""


'''
=====`TRAIN' Usage======
options:

For dse-production (deployed on dev4):
--------------------------------------
python libLinear_MySQL_production.py [params]
	param 0: solver_type
	param 1: start_date
	param 2: end_date
	param 3: dataset_timespan
	param 4: regularization
	param 5: error tolerance
	param 6: bias

-s type : set type of solver (default 1)
	 0 -- L2-regularized logistic regression (primal)
	 1 -- L2-regularized L2-loss support vector classification (dual)
	 2 -- L2-regularized L2-loss support vector classification (primal)
	 3 -- L2-regularized L1-loss support vector classification (dual)
	 4 -- multi-class support vector classification by Crammer and Singer
	 5 -- L1-regularized L2-loss support vector classification
	 6 -- L1-regularized logistic regression
	 7 -- L2-regularized logistic regression (dual)
	11 -- L2-regularized L2-loss epsilon support vector regression (primal)
	12 -- L2-regularized L2-loss epsilon support vector regression (dual)
	13 -- L2-regularized L1-loss epsilon support vector regression (dual)
-c cost : set the parameter C (default 1)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-e epsilon : set tolerance of termination criterion
	-s 0 and 2
		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
		where f is the primal function and pos/neg are # of
		positive/negative data (default 0.01)
	-s 11
		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001) 
	-s 1, 3, 4 and 7
		Dual maximal violation <= eps; similar to libsvm (default 0.1)
	-s 5 and 6
		|f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,
		where f is the primal function (default 0.01)
	-s 12 and 13\n"
		|f'(alpha)|_1 <= eps |f'(alpha0)|,
		where f is the dual function (default 0.1)
-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
-wi weight: weights adjust the parameter C of different classes (see README for details)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)

====`PREDICT' Usage=======

Usage: predict [options] test_file model_file output_file
options:
-b probability_estimates: whether to output probability estimates, 
0 or 1 (default 0); currently for logistic regression only

Note that -b is only needed in the prediction phase. This is different
from the setting of LIBSVM.
'''


import os
import sys
import libLinear_functions
import mysql_lqm
import argparse
import numpy as np

#################################
###     Argument Parser      ####
#################################
parser = argparse.ArgumentParser(description='Perform logistic regression.')
parser.add_argument('--features',        help='Features for logistic regression. e.g. site+ad+site*ad')
parser.add_argument('--start',           help='Start training period. e.g. \'2014-04-01 01:00:00\'')
parser.add_argument('--end',             help='End training period.')
parser.add_argument('--data_dir',        help='Directory where to store the resulting model.')
parser.add_argument('--regularization',  help='Regularization hyperparameter.')
parser.add_argument('--tolerance',       help='Tolerance hyperparameter.')
parser.add_argument('--bias',            help='Bias term w0.')
parser.add_argument('--mysql_host',      help='Mysql host.')
parser.add_argument('--mysql_name',      help='Mysql DB name.')
parser.add_argument('--mysql_user',      help='Mysql user.')
parser.add_argument('--mysql_pass',      help='Mysql password.')
args = parser.parse_args()

#################################
###     DATA GENERATION      ####
#################################
'''Features for LibLinear, '*' creates the interaction'''
features = args.features.split('+')

'''Parameters for the data generation'''
train_per	= [args.start, args.end] #Training Period
rtb_flag	= [1,0] #[1,1]-RTB | [0,0]-Non_RTB | [0,1]or[1,0]-All
# Split crossed features, flatten and remove duplicates.
sql_fields = list(set(sum([fs.split('*') for fs in features], [])))

#################################
###          PATHS           ####
#################################
data_path = args.data_dir + '/'

#  TEMP executable_path = '/opt/liblinear-weights/bin/'
executable_path = '~/projects/dse/lib/liblinear-weights/'

#################################
###       SPARSE MATRIX      ####
#################################
factors = features
non_factors = []
cut_off = 0

#################################
###   LibLINEAR Parameters   ####
#################################
c	= float(args.regularization)   	#regularization
tol	= float(args.tolerance) 	#error uolerance
bias 	= int(args.bias)	#bias term w0
model	= 0 			 					#model selection, check the header
#Note that LIBLINEAR does not use the bias term b by default. If you observe very different results with LIBSVM, 
#try to set -B 1 for LIBLINEAR. This will add the bias term to the loss function as well as the regularization term (w^Tw + b^2). 
#Then, results should be closer.



#################################
###  GETING DATA FROM MySQL  ####
#################################
'''Data is aggregated by features during the query in the MySQL database'''
con_dict={'host':args.mysql_host,
          'db':args.mysql_name,'user':args.mysql_user,'passwd': args.mysql_pass}
sql_table = 'aggregated_ctr' #Data table
train_df = mysql_lqm.MySQL_getdata(con_dict,sql_table,train_per,sql_fields,rtb_flag)

#train_df = mysql_lqm.add_features(train_df)
#factors+=['campaign_id','ad_account_id','pub_account_id', 
#                  'campaign_id*site', 'ad*pub_account_id']
                  
#################################
###  SPARSE MATRIX CREATION  ####
#################################
'''Sparse matrix is created in datapath'''
libLinear_functions.make_sparse(train_df,factors,non_factors,cut_off,data_path)


#################################
###  TRAINING THE MODEL      ####
#################################
'''Removing Previous Model'''
os.system('rm -f '+data_path+'model_summ.model') #Deleting the ex-Model for safety, if train doesnot work it will uses pre-trained model

'''--- TRAINING ---'''
#print (executable_path+'/train -s 0 -c '+str(c)+' -e '+str(tol)+' -B '+str(bias)+' -W '+data_path+'train_ais.txt '+data_path+'train_svm.txt '+data_path+'model_summ.model')
#os.system(executable_path+'/train -s 0 -c '+str(c)+' -e '+str(tol)+' -B '+str(bias)+' -W '+data_path+'train_ais.txt '+data_path+'train_svm.txt '+data_path+'model_summ.model')

model_type=0  #L2 logistic regression
weights_file='train_ais.txt'
train_file='train_svm.txt'
model_file='model_summ.model'
probability_file = 'preds_SummModel_py.txt'

libLinear_functions.fit(executable_path, data_path, train_file,
                    model_file, weights_file, model_type, reg_param=c, tol=tol,
                    has_intercept=bias)


pCTR = libLinear_functions.predict(executable_path, data_path, train_file,
               model_file, probability_file)
if type(pCTR) is not int:
    
    amounts = np.loadtxt(data_path+weights_file)
    y=np.loadtxt(data_path+train_file, usecols=[0])
    # assuming first non-clicks then clicks 
    mean_log_loss, weighted_log_loss = \
                        libLinear_functions.log_loss_weighted(pCTR, y,amounts)
    print mean_log_loss


'''--- LOOKUP ---'''
lookup_file	 = data_path+'lookup.txt'
sc_file	     = data_path+'train_SC'
model_file	 = data_path+'model_summ.model'
libLinear_functions.create_lookup(model_file,sc_file,lookup_file)


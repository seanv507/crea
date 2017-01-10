require (data.table)
require (glmnet)
require(doMC)



setwd("~/projects/dse/dse-production/projects/ctr_data_generation/src")

#('../data/20140623T0100_20140623T0200_ext.csv')
all_data=readMM('../data/20140623T0100_20140623T0200_ext.mtx')
click_no_click<-fread('../data/20140623T0100_20140623T0200_ext_click_no_click.csv')
folds<-fread('../data/20140623T0100_20140623T0200_ext_folds.csv')

#Data
#[1] "timeslot"       "banner_height"  "banner_width"   "site"           "ad"             "clicks"        
#[7] "instances"      "ad_name"        "ad_account_id"  "campaign_id"    "campaign_name"  "site_name"     
#[13] "pub_account_id"


registerDoMC(cores=4)
#system.time(cv_l2<-cv.glmnet(x=all_data,y=as.matrix(click_no_click), foldid=folds,family='binomial',parallel = TRUE, alpha=0))

# system.time(base<-glmnet(x=all_data,y=as.matrix(click_no_click), family='binomial', lambda=0,alpha=0, standardize=FALSE))
# user    system   elapsed 
# 80757.264  1456.166 82375.937 
# Warning messages:
#   1: from glmnet Fortran code (error code -1); Convergence for 1th lambda value not reached after maxit=100000 iterations; solutions for larger lambdas returned 
# 2: In getcoef(fit, nvars, nx, vnames) :
#   an empty model has been returned; probably a convergence issue
# > 80757/60/60
# [1] 22.4325
# > system.time(base_10<-glmnet(x=all_data,y=as.matrix(click_no_click), family='binomial', lambda=10,alpha=0, standardize=FALSE))
# user  system elapsed 
# 4.268   0.735   5.768 




#system.time(cv_l2<-cv.glmnet(x=X,y=as.matrix(click.no_click.amounts), foldid=folds,family='binomial',parallel = TRUE, alpha=0))
# mmdata=readMM('../data/train_sparse.mtx')
# y<-read.table('../data/train_clicks.txt')
# y<-matrix(y$V1)
# weights<-read.table('../data/train_ais.txt')
# weights<-matrix(weights$V1)
# w1=matrix(weights,ncol=2,byrow=FALSE)

#nrow1<-nrow(weights)/2

#mmdata=mmdata[1:nrow1,]


#system.time(l2<-glmnet(mmdata,w1, family='binomial', alpha=0))
# Error in x[o, , drop = FALSE] : incorrect number of dimensions

# system.time(cv_l2<-cv.glmnet(mmdata,w1, family='binomial',parallel = TRUE, alpha=0))
# user   system  elapsed 
# 4229.418  689.056 3751.726

# system.time(cv_l2<-cv.glmnet(mmdata,w1, family='binomial',parallel = FALSE, alpha=0))
# user   system  elapsed 
# 5762.693  945.773 6814.394 



require (glmnet)
require(data.table)
require(plyr)
require(arules)
require(stringr)


y <- matrix(cbind(metrics$clicks - metrics$conversions, metrics$conversions),ncol=2)
# glmnet reports 'response' based on second factor, so put conversions second
y<-matrix(cbind( gp_clicked_names$clicks - gp_clicked_names$conversions,gp_clicked_names$conversions), ncol = 2)
dimnames(y)<-list(NULL,c('no_conversions','conversions'))

i_click <- (metrics$clicks>1)
X <- X[i_click,]
y <- y[i_click,]

# more conversions than clicks
y[y[,2]<0,2] <- 0
y[y[,1]<0,1] <- 0

# 52786 out of 8158991 rows

#cv_base <- cv.glmnet(X,y,family = "binomial",maxit=100000)
# user  system elapsed 
# 155.81   11.69  175.55 
ptm_start <- proc.time()
# was 1000
cv_base_10000 <- cv.glmnet(X,y,family = "binomial",maxit=10000)
ptm_end <- proc.time()
print(ptm_end -  ptm_start)

# user  system elapsed 
# 155.81   11.69  175.55 

# user  system elapsed 
# 182.32   13.42  197.68 100000


# from glmnet Fortran code (error code -2); Convergence for 2th lambda value not reached after maxit=100000 iterations; solutions for larger lambdas returned
# dim(X)
#  1681154  196135
# 10000 iterations
# User      System verstrichen 
# 43213.08       83.59    44430.78
# Warnmeldungen:
#     1: from glmnet Fortran code (error code -22); Convergence for 22th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 2: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 3: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 4: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 5: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 6: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 7: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 8: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 9: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 10: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned
# 11: from glmnet Fortran code (error code -25); Convergence for 25th lambda value not reached after maxit=10000 iterations; solutions for larger lambdas returned

cX <- colSums(X)
# this is wrong ignoring number of clicks

cX <- colSums(gp_clicked_transactions_mat * gp_clicked_names$clicks)
gp_clicked_transactions_mat_500 <- gp_clicked_transactions_mat[,cX >= 500]

# use parallel!!
#sum(cX>=500)
# [1] 8388 1st level interactions
cl<-makeCluster(4)
registerDoSNOW(cl)
ptm_start <- proc.time()
# was 1000
# cv_base_inter_X500_clicks_100000_ using correct 500 clicks thresholding rather than 500 rows as opp to cv_base_inter_X500_clicks_100000_ 9858 vs 8388
cv_base_inter_X500_clicks_100000_ <- cv.glmnet(gp_clicked_transactions_mat_500,y,family = "binomial",maxit=100000, keep=T)
ptm_end <- proc.time()
print(ptm_end -  ptm_start)

ptm_start <- proc.time()
cv_base_inter_X500_clicks_100000_alpha0 <- cv.glmnet(gp_clicked_transactions_mat_500,y,family = "binomial",maxit=100000, keep=T, alpha=0)
print(ptm_end -  ptm_start)
# cv_base_X500_100000 <- cv.glmnet(X_500,y,family = "binomial",maxit=100000)
# User      System verstrichen 
# 3888.19       58.43     4103.17 

# cv_base_inter_X500_100000_ <- cv.glmnet(X_500,y,family = "binomial",maxit=100000)
#8388 columns
# > ptm_end <- proc.time()
# > print(ptm_end -  ptm_start)
# User      System verstrichen 
# 8594.71       63.14     8757.75 
# cv_base_inter_X500_100000_alpha0 <- cv.glmnet(X_500,y,family = "binomial",maxit=100000, alpha=0)
# User      System verstrichen 
# 8161.22      158.14     8672.89 
# cv_base_brand_device <- cv.glmnet(gp_clicked_transactions_brand_device,y,family = "binomial",maxit=100000, keep=T)
# User      System verstrichen 
# 1236.03       59.69     1379.95 

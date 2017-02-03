require (glmnet)
require(data.table)
require(plyr)
require(arules)
require(stringr)
require(ggplot2)
require(tidyr)

z<-coef(cv_base_X500_100000,s='lambda.1se')
z<-coef(cv_base_inter_X500_clicks_100000_,s='lambda.1se')
coefs_df<-data.frame(coef=rownames(z)[z@i+1], val= z[z@i+1]) # THINK need to add 1?

coefs_df[c('coef_grp', 'coef_item')] = str_split_fixed(coefs_df$coef, "=", 2)
a1<-coefs_df[with(coefs_df,order(-abs(val) )),]

ggplot(a1[1:50,],aes(x=reorder(coef,abs(val)), y=val, fill=coef_grp))+geom_bar( stat ='identity') + coord_flip()



z<-coef(cv_base_X500_100000,s='lambda.1se')
z1<-data.frame(coef=rownames(z)[z@i+1], val= z[z@i+1]) # THINK need to add 1?
# can also use predict, type=nonzero - to get nz coeffs



get_crossval_pred_lambda <- function(cv, lambda = c("lambda.1se", "lambda.min"), ...) 
{
    if (is.character(lambda)) {
        lambda = match.arg(lambda)
        lambda = cv[[lambda]]
    }
    i <- which(cv$lambda==lambda)[1]
    cv$fit.preval[,i]
}


gp_clicked_names_pred<-gp_clicked_names

gp_clicked_names_pred['cv_base_brand_device_none_train'] <- 
    predict(cv_base_brand_device,newx=gp_clicked_transactions_brand_device,s=min(cv_base_brand_device$lambda),type='response')

gp_clicked_names_pred['cv_base_brand_device_min_train'] <- 
    predict(cv_base_brand_device,newx=gp_clicked_transactions_brand_device,s='lambda.min',type='response')


gp_clicked_names_pred['cv_base_brand_device_1se_train'] <- 
    predict(cv_base_brand_device,newx=gp_clicked_transactions_brand_device,s='lambda.1se',type='response')

gp_clicked_names_pred['cv_base_brand_device_min_xval'] <- 
    get_crossval_pred_lambda(cv_base_brand_device,'lambda.min')
gp_clicked_names_pred['cv_base_brand_device_1se_xval'] <- 
    get_crossval_pred_lambda(cv_base_brand_device,'lambda.1se')




gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_min_train'] <- 
    predict(cv_base_inter_X500_clicks_100000_,newx=gp_clicked_transactions_mat_500,s='lambda.min',type='response')
gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_1se_train'] <- 
    predict(cv_base_inter_X500_clicks_100000_,newx=gp_clicked_transactions_mat_500,s='lambda.1se',type='response')
gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_alpha0_min_train'] <- 
    predict(cv_base_inter_X500_clicks_100000_alpha0,newx=gp_clicked_transactions_mat_500,s='lambda.min',type='response')
gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_alpha0_1se_train'] <- 
    predict(cv_base_inter_X500_clicks_100000_alpha0,newx=gp_clicked_transactions_mat_500,s='lambda.1se',type='response')



gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_min_xval'] <- 
    get_crossval_pred_lambda(cv_base_inter_X500_clicks_100000_,'lambda.min')
gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_1se_xval'] <- 
    get_crossval_pred_lambda(cv_base_inter_X500_clicks_100000_,'lambda.1se')
gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_alpha0_min_xval'] <- 
    get_crossval_pred_lambda(cv_base_inter_X500_clicks_100000_alpha0,'lambda.min')
gp_clicked_names_pred['cv_base_inter_X500_clicks_100000_alpha0_1se_xval'] <- 
    get_crossval_pred_lambda(cv_base_inter_X500_clicks_100000_alpha0,'lambda.1se')




save(gp_clicked_names_pred, file='gp_clicked_names_pred')
load('../data/out/gp_clicked_names_pred')
qu<-wtd.quantile(gp_clicked_names_pred$cv_base_inter_X500_clicks_100000_min_train,
                 weights=gp_clicked_names_pred$clicks,seq(0,1,.1))
qu<-round(qu*100,1)
# identify cuts eg .5% to 2.5% step .25%
# identify outliers!!!

gp_clicked_names_pred_dt<-as.data.table(gp_clicked_names_pred)

fac_names <-lapply(gp_clicked_names_pred,class)
fac_names <- names(fac_names[fac_names=='factor'])
uni_names <- c(fac_names, 'week') 

preds<-c('cv_base_inter_X500_clicks_100000_min_train','cv_base_inter_X500_clicks_100000_min_xval',
         'cv_base_brand_device_min_xval','cv_base_inter_X500_clicks_100000_min_train',
         'cv_base_brand_device_none_train')



cvr_dts <- sapply( uni_names, 
                   function (fac) describe_factor_means(gp_clicked_names_pred_dt, 'clicks', 'conversions', 
                                                        fac, means = preds, conf_int = 0.95), simplify = F)
cvr_dts_20 <- sapply(cvr_dts, function (dt) dt[1:min(20,.N),], simplify=F)


plot_cvr_facet <- function (dt) {
  
    ggplot(dt, aes(x=reorder(fact, rate_l), y=rate, fill=fact)) + 
        xlab(dt$group) + 
        geom_bar(stat = 'identity') + 
        geom_errorbar(aes(ymin=rate_l, ymax=rate_u) ) +
        guides(fill=FALSE) + 
        ylim(0,10) +
        coord_flip() +
        facet_wrap(~group, nrow=5, ncol=4, scales="free", dir="v")
    # problems with variable column name
}


plot_cvr_a <- function (dt) {
    # lb<-dt$group[1]
    ggplot(dt, aes(x=reorder(fact, rate_l), y=rate, fill=fact)) + 
        xlab(dt$group) +
        geom_bar(stat = 'identity') + 
        geom_errorbar(aes(ymin=rate_l, ymax=rate_u) ) +
        guides(fill=FALSE) + 
        ylim(0,5) +
        coord_flip()
    # problems with variable column name
}    

plot_cvr_a(cvr_dts_20$week)

p1<- plot_cvr_a(b2)
dat1<- gather_(b2,'model','prediction',preds)
p1 + geom_point(data=dat1, size=2,aes(y=prediction,colour=model, shape=model))

cvr_dts_20_plots <- sapply(cvr_dts_20, plot_cvr_a, simplify = F)

save.graph <- function (x) ggsave(paste0('cvr_',names(x)),plot=x,device='pdf',width=4,units='in')

sapply(names(cvr_dts_20_plots), function (fac) ggsave(paste0('cvr_',fac),plot=cvr_dts_20_plots[[fac]],device='pdf',width=4,units='in'),simplify=F)

dat1<- gather_(b2,'model','prediction',preds)
cvr_dts_20_all <-rbindlist(cvr_dts_20)
p2<- plot_cvr_a(cvr_dts_20_all)
dat2<- gather_(cvr_dts_20_all,'model','prediction',preds)
p3<-p2 + 
    #xlab(group) + 
    facet_wrap(~group, nrow=5, ncol=4, scales="free", dir="v")

# save eachgroup separately then rbind [how to plot all together?]
a1<-gp_clicked_names_pred_dt[,.(actual=100*sum(conversions)/sum(clicks),
                                cv_base_inter_X500_clicks_100000_min_xval=100*weighted.mean(cv_base_inter_X500_clicks_100000_min_xval, w=clicks)),
                             by=cut(cv_base_inter_X500_clicks_100000_min_train*100,seq(0,4,0.25))]
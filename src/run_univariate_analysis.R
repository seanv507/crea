require(SOAR)
require(data.table)
require(ggplot2)
require(svglite)

src_dir <- "~/projects/crealytics/src"
data_dir <- "~/projects/crealytics/data"
setwd(src_dir)


# univariate analysis
if (!file.exists(paste(data_dir,'clicked_categories.csv',sep='/'))){
  cats<- gp_clicked_names_mapped_dt[,.N,by=.(category_1st_level,category_2nd_level,category_3rd_level,category_4th_level,category_5th_level,item_id)]
  fwrite(cats, paste(data_dir,'clicked_categories.csv',sep='/'))
}  

if (!file.exists(paste(data_dir,'clicked_product_types.csv',sep='/'))){
  products<- gp_clicked_names_mapped_dt[,.N,by=.(product_type_1st_level,product_type_2nd_level,product_type_3rd_level,product_type_4th_level,product_type_5th_level,item_id)]
  fwrite(products,paste(data_dir,'clicked_product_types.csv',sep='/'))
}


describe_factor_means<-function (dt, count, event, fact, conf_int = 0.95, means = NULL){
  # means is for calculating mean prediction for factor instance ( taking into account number of samples)
  # fact can be a comma separated string
  if (is.null(means)){
    gp_dt<-dt[,.(
      count = sum(get(count)),
      rate = 100*sum(get(event))/sum(get(count)),
      rate_l = 100*qbeta((1 - conf_int) / 2, sum(get(event)) + .5, sum(get(count)) - sum(get(event)) + .5),
      rate_u = 100*qbeta((1 + conf_int) / 2, sum(get(event)) + .5, sum(get(count)) - sum(get(event)) + .5)
    ),
    by = fact][order(-count)]
  }else{
    gp_dt<-dt[,c(.(
      count = sum(get(count)),
      rate = 100*sum(get(event))/sum(get(count)),
      rate_l = 100*qbeta((1 - conf_int) / 2, sum(get(event)) + .5, sum(get(count)) - sum(get(event)) + .5),
      rate_u = 100*qbeta((1 + conf_int) / 2, sum(get(event)) + .5, sum(get(count)) - sum(get(event)) + .5)),
      #lapply(.SD,function(x) 100*weighted.mean(x=x, w = get(count))) [was using probs instead of w gave wrong answers !?!
      lapply(.SD,function(x) 100*sum(x * get(count))/sum(get(count)))
    ),
    by = fact, .SDcols=means][order(-count)]
    # note you have to c() the two lists together http://stackoverflow.com/a/20460441
  }
  setnames(gp_dt,fact, 'fact')
  gp_dt[['group']] = fact #standardise name for concatenation/ggplot..
  gp_dt
}




fac_names <-lapply(gp_clicked_names_mapped,class)
fac_names <- names(fac_names[fac_names=='factor'])
uni_names <- c(fac_names, 'week') 

# a<-describe_factor_means(gp_clicked_names_mapped_dt, 'clicks', 'conversions', fact = 'day_week')
# a<-describe_factor_means(gp_clicked_names_mapped_dt, 'clicks', 'conversions', fact = 'day_week', means=c('cv_base_inter_X500_clicks_100000_min_train'))
# b<-describe_factor(gp_clicked_names_mapped_dt, 'clicks', 'conversions', fact = 'brand,category_1st_level')

if (!'cvr_dts' %in% Ls()){
  cvr_dts <- sapply( uni_names, function (fac) describe_factor_means(gp_clicked_names_mapped_dt, 'clicks', 'conversions', fac), simplify = F)
  cvr_dts_20 <- sapply(cvr_dts, function (dt) dt[1:min(20,.N),], simplify=F)
  Store(cvr_dts, cvr_dts_20)
}
# http://docs.ggplot2.org/current/aes_.html
plot_cvr <- function (dt) {
  ggplot(dt, aes(x=reorder(dt[[colnames(dt)[1]]], dt[['rate_l']]), y=rate, fill=dt[[colnames(dt)[1]]])) + 
    xlab(dt$group) + 
    geom_bar(stat = 'identity') + 
    geom_errorbar(aes(ymin=rate_l, ymax=rate_u) ) +
    guides(fill=FALSE) + 
    ylim(0,5) +
    coord_flip()
}

plot_cvr_a <- function (dt) {
  ggplot(dt, aes(x=dt[[colnames(dt)[1]]], y=rate, fill=dt[[colnames(dt)[1]]])) + 
    xlab(dt$group) + 
    geom_bar(stat = 'identity') + 
    geom_errorbar(aes(ymin=rate_l, ymax=rate_u) ) +
    guides(fill=FALSE) + 
    ylim(0,5) +
    coord_flip()
}
# distinguish between ordered factor and reorder if not..

for (f in uni_names){
  b <- plot_cvr(cvr_dts_20[[f]])
  ggsave(file=paste0(data_dir,'/', 'CVR_',f,'.pdf'),plot=b, device='pdf',width=4,units='in')
  ggsave(file=paste0(data_dir,'/', 'CVR_',f,'.svg'),plot=b, device='svg',width=4,units='in')
}

date_names=c('day_week','month_f','week')
# don't reorder and have more than 20 cats (for week)
for (f in date_names){
  b <- plot_cvr_a(cvr_dts[[f]])
  ggsave(file=paste0(data_dir,'/', 'CVR_',f,'.pdf'),plot=b, device='pdf',width=4,units='in')
  ggsave(file=paste0(data_dir,'/', 'CVR_',f,'.svg'),plot=b, device='svg',width=4,units='in')
}
# libreoffice doesn't show svg?

plot_cvr_facet <- function (dt) {
  ggplot(dt, aes(x=reorder(fact, rate_l), y=rate, fill=fact)) + 
    xlab(group) + 
    geom_bar(stat = 'identity') + 
    geom_errorbar(aes(ymin=rate_l, ymax=rate_u) ) +
    guides(fill=FALSE) + 
    ylim(0,10) +
    coord_flip() +
    facet_wrap(~group, nrow=5, ncol=4, scales="free", dir="v")
  # problems with variable column name
}

plot_clicks_facet <- function (dt) {
  ggplot(dt, aes(x=reorder(fact, count), y=count, fill=fact)) + 
    geom_bar(stat = 'identity') + 
    guides(fill=FALSE) + 
    scale_y_log10()  +
    coord_flip() +
    facet_wrap(~group, nrow=5, ncol=4, scales="free", dir="v")
  # problems with variable column name
}

plot_hist_facet <- function (dt) {
  ggplot(dt, aes(x=count)) + 
    geom_bar() + 
    stat_bin(binwidth=100) +
    scale_x_log10()  +
    facet_wrap(~group, nrow=5, ncol=4, scales="free", dir="v")
  # problems with variable column name
}

cvr_dts_all <- rbindlist(cvr_dts)

plot_cvr_facet(cvr_dts_20_all)
plot_clicks_facet(cvr_dts_20_all)
plot_hist_facet(cvr_dts_all)

cvr_dts_20_plots <- sapply(cvr_dts_20, plot_cvr, simplify=F)
# dt[,1] didn't work ... even on reorder

multiplot(plotlist = cvr_dts_20_plots[1:12], cols=3)

ggplot(a, aes(x=day_week, y=rate)) + geom_bar(stat = 'identity') + geom_errorbar(aes(ymin=rate_l, ymax=rate_u))
# can reorder by value
# ggplot(a, aes(x=reorder(day_week,rate_l), y=rate)) + geom_bar(stat = 'identity') + geom_errorbar(aes(ymin=rate_l, ymax=rate_u))

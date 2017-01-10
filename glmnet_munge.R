require (data.table)
require (glmnet)
require(doMC)

setwd("~/projects/dse/dse-production/projects/ctr_data_generation/src")
data<- fread('../data/20140623T0100_20140623T0200_ext.csv') # 30 times faster
log.loss<-function (targets,predicts){
  # we assume targets is 2 column amounts
  -(targets$clicks * log(predicts) + targets$non.clicks*log(1-predicts))/sum(targets)
}

# do arithmetic before convert to factors
data.factors=c("banner_height","banner_width","site","ad","ad_name" ,"ad_account_id", "campaign_id","campaign_name","site_name" )
for (fac in data.factors) {
  set(data, j=fac, value=as.factor(data[[fac]]))
}

y<-data.frame(non.clicks=data$instances-data$clicks,clicks=data$clicks)
folds<-data$timeslot+1


#Data
#[1] "timeslot"       "banner_height"  "banner_width"   "site"           "ad"             "clicks"        
#[7] "instances"      "ad_name"        "ad_account_id"  "campaign_id"    "campaign_name"  "site_name"     
#[13] "pub_account_id"

factors=c('site','ad', 'site,ad')
fac_splits=strsplit(factors,',')
fac_ranks=list()
col_list=list()
fac_ranks<-lapply(factors,function (fac) data[,list(sum(instances),sum(clicks)),by=fac]) 
for (i in 1:length(factors)){
#  fac_ranks[i]=data[,list(sum(instances),sum(clicks)),by=factors[i]] didn't work
  setnames(fac_ranks[[i]],"V1","instances")
  setnames(fac_ranks[[i]],"V2","clicks")
  fac_ranks[[i]][,ranking:=rank(-instances,ties.method="first")]
  fac_ranks[[i]][,m25:=pbeta(clicks/instances*.75,clicks+0.5,instances-clicks+0.5)] # 
  fac_ranks[[i]][,p25:=pbeta(clicks/instances*1.25,clicks+0.5,instances-clicks+0.5)]
  fac_ranks[[i]][,p25_m25:=p25-m25]
  fac_ranks[[i]][,est_inst:=(p25_m25>0.95)*instances]
  fac_ranks[[i]][,sum(instances)]
  fac_ranks[[i]][,sum(est_inst)]
  print(fac_ranks[[i]][,sum(est_inst)]/fac_ranks[[i]][,sum(instances)])
  
  setkeyv(fac_ranks[[i]],cols=fac_splits[[i]])
  setkeyv(data,cols=fac_splits[[i]])
  col_list[[i]]=fac_ranks[[i]][data,ranking-1]
}
#z2<-data[,list(sum(instances),sum(clicks)),by='site,ad']

# setnames(z2,"V1","instances")
# setnames(z2,"V2","clicks")
# setkeyv(z2,cols=c('site','ad')) # should be possible to do in by calculation?
# 
# 
# z2[,ranking:=rank(-instances,ties.method="first")]
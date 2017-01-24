require(arules)


fac_names <-lapply(gp_clicked_names_dt,class)
fac_names <- names(fac_names[fac_names=='factor'])
print(fac_names)
# nrow(gp_clicked)
# [1] 1681154
# 0.0005948295

X_names <- colnames(gp_clicked_names_dt)
X_names <- X_names[!(X_names %in% c('week', 'month','impressions','conversions','clicks','cross_device_conv','total_conv_value','cost'))]
a1 <- as.data.frame(gp_clicked_names_dt[,mget(X_names)])

a1_char <- sapply(gp_clicked_names_dt,class)
a1_char <- names(a1_char[a1_char=='character'])
#a1[a1_char] <- sapply(a1[a1_char],as.factor)
for (f in a1_char){
    gp_clicked_names_dt[[f]]<- as.factor(gp_clicked_names_dt[[f]])
}

gp_clicked_transactions <- as(gp_clicked_names_dt[,fac_names, with=F] , 'transactions')

gp_clicked_transactions_mat <- t(as(gp_clicked_transactions,'ngCMatrix'))
# for glmnet

n_occur <- 1000
frequent <- apriori(gp_clicked_transactions, parameter = list(support= n_occur/length(gp_clicked_transactions), target="frequent", minlen=2, maxtime=30))
n_occur <- 10000
frequent_10000 <- apriori(gp_clicked_transactions, parameter = list(support= n_occur/length(gp_clicked_transactions),
                                                                    target="frequent", minlen=1, maxtime=30))
summary(frequent_10000)


inspect(head(subset(frequent_10000, subset = size(items)==10),n=10, by='support'))

length(levels(gp_clicked_names$item_id))

fac_names_no_cats <-c("brand", "ad_group",  "campaign_id" ,"item_id", "day_week", "device")
gp_clicked_no_cats_transactions <- as(gp_clicked_names[fac_names], 'transactions')

fac_names_cats <- c("item_id", "category_1st_level", "category_2nd_level", "category_3rd_level", "category_4th_level", "category_5th_level",
"product_type_1st_level", "product_type_2nd_level", "product_type_3rd_level", "product_type_4th_level", "product_type_5th_level")

categories_dt <- as.data.table(gp_clicked_names[fac_names_cats])
# http://stackoverflow.com/a/8212756/
# we just select 1 if mutiple cats for same itemid
categories_dt_unique <- categories_dt[,.SD[1], by='item_id']
categories_dt_unique_label <- categories_dt_unique[, item_id := paste0('item_id=',item_id)]
setnames(categories_dt_unique_label, 'item_id', 'labels')



gp_clicked_no_cats_transactions@itemInfo <-categories_dt_unique_label

gp_clicked_no_cats_transactions_multilevel <- addAggregate(gp_clicked_no_cats_transactions, "product_type_1st_level")


# Error in .local(x, ...) : 
#     Name not available in itemInfo or supplied number of by does not match number of items in x!
#     > 

# need to add aggregate for each 

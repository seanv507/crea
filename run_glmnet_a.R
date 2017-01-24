require (glmnet)
require(data.table)
require(plyr)
require(arules)

gp_clicked<- read.csv('gp_clicked_20160201_20161201.csv')

lookup_sql = 'select * from lookup_20160201_20161201'
lookup_df = sqldf(lookup_sql, dbname = 'crealytics.db')

orig_names <- c('Ad group', 'Ad group ID', 'Brand', 'Campaign', 'Campaign ID', 'Category (1st level)', 'Category (2nd level)', 'Category (3rd level)', 'Category (4th level)', 'Category (5th level)',
              'Country/Territory', 'Custom label 0', 'Custom label 1', 'Custom label 2', 'Custom label 3', 'Custom label 4', 'Device', 'Item Id', 
              'Product type (1st level)',  'Product type (2nd level)', 'Product type (3rd level)', 'Product type (4th level)', 'Product type (5th level)')
std_names <- c('ad_group', 'ad_group_id', 'brand', 'campaign', 'campaign_id', 'category_1st_level',   'category_2nd_level',   'category_3rd_level',    'category_4th_level', 'category_5th_level',
             'country', 'custom_label_0', 'custom_label_1', 'custom_label_2', 'custom_label_3', 'custom_label_4', 'device', 'item_id', 
             'product_type_1st_level', 'product_type_2nd_level', 'product_type_3rd_level', 'product_type_4th_level', 'product_type_5th_level')
rename_cols <- data.frame(orig = orig_names, std = std_names,stringsAsFactors = F)

map_names <- function(df, lookup, names_df){
    
    df_1 <- df
    for (i in seq(nrow(names_df))) {
        orig <- names_df[[i, 'orig']]
        std <-  names_df[[i, 'std']]
        if (std %in% names(df)){
            print(i)
            lo <- lookup[lookup$id_name == orig, c('id','id_index')]
            mapped <- suppressWarnings(mapvalues(df_1[[std]], from = lo$id_index, to = lo$id))
            df_1[[std]] <- as.factor(mapped)
            
            df_1$id <- NULL
        }
    }
    df_1
}

#d <- gp_clicked[sample(nrow(gp_clicked),100),]
gp_clicked_names <- map_names(gp_clicked, lookup_df, rename_cols)
save(gp_clicked_names, file='gp_clicked_names')

load( file='gp_clicked_names')


gp_clicked_names['day_week_name']<-mapvalues(gp_clicked_names$day_week, from = seq(0,6), 
                                             to =c('sun','mon','tue','wed','thu','fri','sat'))
dw <-as.factor(gp_clicked_names$day_week_name)
dw<-NULL


sqldf1 <- sqldf("SELECT lookup_df.
FROM largetable
                INNER JOIN lookup
                ON largetable.HouseType = lookup.HouseType")

X <- readMM('C:/Users/Sean Violante/Documents/Projects/crealytics/data_201602_201603.mtx')
metrics <- read.csv('metrics_201602_201603.csv')


y <- matrix(cbind(metrics$conversions, metrics$clicks - metrics$conversions),ncol=2)
dimnames(y)<-list(NULL,c('conversions','no_conversions'))
i_click <- (metrics$clicks>1)
y[y[,2]<0,2] <- 0

X <- X[i_click,]
y <- y[i_click,]
# 52786 out of 8158991 rows

#cv_base <- cv.glmnet(X,y,family = "binomial",maxit=100000)
# user  system elapsed 
# 155.81   11.69  175.55 
ptm_start <- proc.time()
cv_base <- cv.glmnet(X,y,family = "binomial",maxit=1000)
ptm_end <- proc.time()
print(ptm_end -  ptm_start)

> # user  system elapsed 
    > # 155.81   11.69  175.55 
    > ptm_end <- proc.time()
# user  system elapsed 
# 182.32   13.42  197.68 100000


# from glmnet Fortran code (error code -2); Convergence for 2th lambda value not reached after maxit=100000 iterations; solutions for larger lambdas returned



fac_names <-lapply(gp_clicked_names,class)
fac_names <- names(fac_names[fac_names=='factor'])
print(fac_names)
# nrow(gp_clicked)
# [1] 1681154
# 0.0005948295
gp_clicked_transactions <- as(gp_clicked_names[fac_names], 'transactions')

gp_clicked_transactions_mat <- t(as(gp_clicked_transactions,'ngCMatrix'))

y<-matrix(cbind(gp_clicked_names$conversions, gp_clicked_names$clicks - gp_clicked_names$conversions), ncol = 2)


frequent <- apriori(gp_clicked_transactions, parameter = list(support= 0.0006, target="frequent", minlen=2, maxtime=30))

summary(frequent)

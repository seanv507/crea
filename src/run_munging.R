require(data.table)
require(plyr)
require(sqldf)
require(ggplot2)
require(stringr)
require(Hmisc)
require(SOAR)
# to do

# lay out whole pipeline 
# split charts ? one per page?

#do results by month too
src_dir <- "~/projects/crealytics/src"
data_dir <- "~/projects/crealytics/data"
setwd(src_dir)

Sys.setenv(R_LOCAL_LIB_LOC = data_dir)

# get the data from database .  taking only clicked data uses much less memory
if (!'gp_clicked_names' %in% Ls()){
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
  gp_clicked_names <- map_names(gp_clicked, lookup_df, rename_cols)
  # load(paste(data_dir,'gp_clicked_names.RData',sep='/'))
  #save(gp_clicked_names, file='gp_clicked_names.RData')
  Store(gp_clicked_names)  
}

if (!'gp_clicked_names_mapped' %in% Ls()){
  #load( file=paste(data_dir,'gp_clicked_names_mapped.RData',sep='/'))
  #gp_clicked_names_mapped <- gp_clicked_names
  
    
  cat_names <- c('category_1st_level',   'category_2nd_level',   'category_3rd_level',    'category_4th_level', 'category_5th_level')
  prod_names <- c('product_type_1st_level', 'product_type_2nd_level', 'product_type_3rd_level', 'product_type_4th_level', 'product_type_5th_level')
  
  # add higher level names so distinguish '--' of differeent higher level categories 
  replace_blank <- function(df, nam){
      for (i in seq(length(nam) - 1)){
          is_blank <- df[nam[i + 1]] =='--'
          renamed_levels <- paste('--', df[is_blank, nam[i]])
          levels(df[[nam[ i + 1]]]) <- c(levels(df[[nam[i+1]]]), unique(renamed_levels))
          # so we are still keeping '--' level... OK?
          
          df[is_blank, nam[i+1]] <- renamed_levels
      }
      df
  }
  gp_clicked_names_mapped <- gp_clicked_names
  gp_clicked_names_mapped <- replace_blank( gp_clicked_names_mapped, cat_names)
  gp_clicked_names_mapped <- replace_blank( gp_clicked_names_mapped, prod_names)

  gp_clicked_names_mapped['day_week']<-mapvalues(gp_clicked_names_mapped$day_week, from = seq(0,6), 
                                          to =c('sun','mon','tue','wed','thu','fri','sat'))
  # checked date mapping
  gp_clicked_names_mapped$day_week <-factor(gp_clicked_names_mapped$day_week, 
                                     levels= c('mon','tue','wed','thu','fri','sat', 'sun'))

  gp_clicked_names_mapped['month_f']<-factor(gp_clicked_names_mapped$month, 
                                      levels=c('1','2','3','4','5','6','7','8','9','10','11','12'),
                                      labels=c('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'))

  #load( file=paste(data_dir,'gp_clicked_names_mapped.RData',sep='/'))
  #save(gp_clicked_names, file='gp_clicked_names_mapped')
  # note original script had mapped data under same name (gp_clicked_names) 
  Store(gp_clicked_names_mapped)
}



gp_clicked_names_mapped_dt <- as.data.table(gp_clicked_names_mapped)

make_interaction_factor_2D <- function(dt,fac1, fac2){
    factors=c(fac1,fac2)
    inter_name <- paste(factors, collapse = '*')
    print( inter_name)
    dt[,eval(inter_name) := paste(get(fac1), get(fac2),sep='*')]
    dt
}
# couldn't find way of creatinbg arbitrary interaction order
interaction_factor <- function(dt,factors){
    #inter_name <- 
    args <- as.list(factors)
    args$sep='*'
    dt[,eval(inter_name) := do.call(paste,args,quote=T)]
    # do.call
    dt
}

# df <- data.frame(h1=c('b','c'), h2=c('e','f'),g=c(1,2))
# dt <- as.data.table(df)
# interaction_factor(dt, c('h1', 'h2'))
# make_interaction_factor_2D(dt, 'h1', 'h2')
# dt



# 
# fac_names
# [1] "ad_group_id"            "ad_group"               "campaign_id"            "day_week"               "device"                
# [6] "category_1st_level"     "category_2nd_level"     "category_3rd_level"     "category_4th_level"     "category_5th_level"    
# [11] "product_type_1st_level" "product_type_2nd_level" "product_type_3rd_level" "product_type_4th_level" "product_type_5th_level"
# [16] "brand"                  "item_id"                "month_f"   

make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'brand', 'category_1st_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'brand', 'category_2nd_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'brand', 'category_3rd_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'brand', 'category_4th_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'brand', 'category_5th_level')

make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'month_f', 'category_1st_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'month_f', 'category_2nd_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'month_f', 'category_3rd_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'month_f', 'category_4th_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'month_f', 'category_5th_level')

make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'device', 'brand')

make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'device', 'category_1st_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'device', 'category_2nd_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'device', 'category_3rd_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'device', 'category_4th_level')
make_interaction_factor_2D(gp_clicked_names_mapped_dt, 'device', 'category_5th_level')


# DOUBLE CHECK MAPPING!!
# X <- readMM('C:/Users/Sean Violante/Documents/Projects/crealytics/data_201602_201603.mtx')
# metrics <- read.csv('metrics_201602_201603.csv')




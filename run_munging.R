require(data.table)
require(plyr)
require(sqldf)
require(ggplot2)
require(stringr)
require(Hmisc)

# Multiple plot function
# http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    requre(grid)
    
    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)
    
    numPlots = length(plots)
    
    # If layout is NULL, then use 'cols' to determine layout
    if (is.null(layout)) {
        # Make the panel
        # ncol: Number of columns of plots
        # nrow: Number of rows needed, calculated from # of cols
        layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                         ncol = cols, nrow = ceiling(numPlots/cols))
    }
    
    if (numPlots==1) {
        print(plots[[1]])
        
    } else {
        # Set up the page
        grid.newpage()
        pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
        
        # Make each plot, in the correct location
        for (i in 1:numPlots) {
            # Get the i,j matrix positions of the regions that contain this subplot
            matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
            
            print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                            layout.pos.col = matchidx$col))
        }
    }
}

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

#save(gp_clicked_names, file='gp_clicked_names')

cat_names <- c('category_1st_level',   'category_2nd_level',   'category_3rd_level',    'category_4th_level', 'category_5th_level')
prod_names <- c('product_type_1st_level', 'product_type_2nd_level', 'product_type_3rd_level', 'product_type_4th_level', 'product_type_5th_level')

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

gp_clicked_names <- replace_blank( gp_clicked_names, cat_names)
gp_clicked_names <- replace_blank( gp_clicked_names, prod_names)

gp_clicked_names['day_week']<-mapvalues(gp_clicked_names$day_week, from = seq(0,6), 
                                        to =c('sun','mon','tue','wed','thu','fri','sat'))
# checked date mapping
gp_clicked_names$day_week <-factor(gp_clicked_names$day_week, 
                                   levels= c('mon','tue','wed','thu','fri','sat', 'sun'))

gp_clicked_names['month_f']<-factor(gp_clicked_names$month, 
                                    levels=c('1','2','3','4','5','6','7','8','9','10','11','12'),
                                    labels=c('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'))





save(gp_clicked_names, file='gp_clicked_names_mapped')

load( file='gp_clicked_names_mapped')

gp_clicked_names_dt <- as.data.table(gp_clicked_names)

make_interaction_factor_2D <- function(dt,fac1, fac2){
    factors=c(fac1,fac2)
    inter_name <- paste(factors, collapse = '*')
    print( inter_name)
    dt[,eval(inter_name) := paste(get(fac1), get(fac2),sep='*')]
    dt
}

interaction_factor <- function(dt,factors){
    #inter_name <- 
    args <- as.list(factors)
    args$sep='*'
    dt[,eval(inter_name) := do.call(paste,args,quote=T)]
    # do.call
    dt
}

df <- data.frame(h1=c('b','c'), h2=c('e','f'),g=c(1,2))
dt <- as.data.table(df)
interaction_factor(dt, c('h1', 'h2'))
make_interaction_factor_2D(dt, 'h1', 'h2')
dt

#interaction_factor(gp_clicked_names_dt, c('brand', 'category_1st_level'))


# 
# fac_names
# [1] "ad_group_id"            "ad_group"               "campaign_id"            "day_week"               "device"                
# [6] "category_1st_level"     "category_2nd_level"     "category_3rd_level"     "category_4th_level"     "category_5th_level"    
# [11] "product_type_1st_level" "product_type_2nd_level" "product_type_3rd_level" "product_type_4th_level" "product_type_5th_level"
# [16] "brand"                  "item_id"                "month_f"   

make_interaction_factor_2D(gp_clicked_names_dt, 'brand', 'category_1st_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'brand', 'category_2nd_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'brand', 'category_3rd_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'brand', 'category_4th_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'brand', 'category_5th_level')

make_interaction_factor_2D(gp_clicked_names_dt, 'month_f', 'category_1st_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'month_f', 'category_2nd_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'month_f', 'category_3rd_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'month_f', 'category_4th_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'month_f', 'category_5th_level')

make_interaction_factor_2D(gp_clicked_names_dt, 'device', 'brand')

make_interaction_factor_2D(gp_clicked_names_dt, 'device', 'category_1st_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'device', 'category_2nd_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'device', 'category_3rd_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'device', 'category_4th_level')
make_interaction_factor_2D(gp_clicked_names_dt, 'device', 'category_5th_level')



# univariate analysis
cats<- gp_clicked_names_dt[,.N,by=.(category_1st_level,category_2nd_level,category_3rd_level,category_4th_level,category_5th_level,item_id)]
products<- gp_clicked_names_dt[,.N,by=.(product_type_1st_level,product_type_2nd_level,product_type_3rd_level,product_type_4th_level,product_type_5th_level,item_id)]
fwrite(cats,'clicked_categories.csv')
fwrite(products,'clicked_product_types.csv')



describe_factor<-function (dt, count, event, fact, conf_int = 0.95){
    # fact can be a comma separated string 
    gp_dt<-dt[,.(
          count = sum(get(count)), 
          rate = 100*sum(get(event))/sum(get(count)),
          rate_l = 100*qbeta((1 - conf_int) / 2, sum(get(event)) + .5, sum(get(count)) - sum(get(event)) + .5),
          rate_u = 100*qbeta((1 + conf_int) / 2, sum(get(event)) + .5, sum(get(count)) - sum(get(event)) + .5)
          ), 
       by = fact][order(-count)]
    #setnames(gp_dt,fact, 'fact')
    gp_dt[['group']] = fact #standardise name for concatenation/ggplot..
    gp_dt
}



describe_factor_means<-function (dt, count, event, fact, means = NULL, conf_int = 0.95){
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


fac_names <-lapply(gp_clicked_names,class)
fac_names <- names(fac_names[fac_names=='factor'])
uni_names <- c(fac_names, 'month','week') 

a<-describe_factor(gp_clicked_names_dt, 'clicks', 'conversions', fact = 'day_week')
a<-describe_factor_means(gp_clicked_names_dt, 'clicks', 'conversions', fact = 'day_week', means=c('cv_base_inter_X500_clicks_100000_min_train'))
b<-describe_factor(gp_clicked_names_dt, 'clicks', 'conversions', fact = 'brand,category_1st_level')
cvr_dts <- sapply( uni_names, function (fac) describe_factor(gp_clicked_names_dt, 'clicks', 'conversions', fac), simplify = F)
cvr_dts_20 <- sapply(cvr_dts, function (dt) dt[1:min(20,.N),], simplify=F)
# http://docs.ggplot2.org/current/aes_.html
plot_cvr <- function (dt) {
    ggplot(dt, aes(x=reorder(dt[[colnames(dt)[1]]], dt[['rate_l']]), y=rate, fill=dt[[colnames(dt)[1]]])) + 
    xlab(group) + 
    geom_bar(stat = 'identity') + 
    geom_errorbar(aes(ymin=rate_l, ymax=rate_u) ) +
    guides(fill=FALSE) + 
    ylim(0,10) +
    coord_flip()
}

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

DOUBLE CHECK MAPPING!!
X <- readMM('C:/Users/Sean Violante/Documents/Projects/crealytics/data_201602_201603.mtx')
metrics <- read.csv('metrics_201602_201603.csv')




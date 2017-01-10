# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 23:55:38 2017

@author: Sean Violante
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
import scipy.sparse
from scipy.stats import beta
import matplotlib.pyplot as plt
import sqlalchemy 
import crealytics
import sparsecat

engine = sqlalchemy.create_engine('sqlite:///crealytics.db')
data_sql="""
select 
     ad_group_id,
     ad_group,
     campaign_id,
     strftime('%W',day) week,
     strftime('%m',day) month,
     strftime('%w',day) day_week,
     device,
     category_1st_level,
     category_2nd_level,
     category_3rd_level,
     category_4th_level,
     category_5th_level,
     product_type_1st_level,
     product_type_2nd_level,
     product_type_3rd_level,
     product_type_4th_level,
     product_type_5th_level,
     brand,
     item_id,
     sum(impressions) impressions,
     sum(clicks) clicks,
     sum(conversions) conversions,
     sum(cross_device_conv) cross_device_conv,
     sum(total_conv_value) total_conv_value,
     sum(cost) cost
     
from 
    report_20160201_20161201
where
    day < julianday('2016-04-01')
group by 
     ad_group_id,
     ad_group,
     campaign_id,
     week,
     month,
     day_week,
     device,
     category_1st_level,
     category_2nd_level,
     category_3rd_level,
     category_4th_level,
     category_5th_level,
     product_type_1st_level,
     product_type_2nd_level,
     product_type_3rd_level,
     product_type_4th_level,
     product_type_5th_level,
     brand,
     item_id
"""
# %matplotlib

lookup_sql = 'select * from lookup_20160201_20161201'
lookup_df = pd.read_sql_query(lookup_sql, engine)
lookup_df.set_index(['id_name','id_index'], inplace=True)

#%time 
# data_df=pd.read_sql_query(data_sql,engine)
# Wall time: 15min 26s
data_df=pd.read_pickle('report_group_201602_201603.pkl')

#TODO replace '-- ' with something specific to higher category
#TODO what to do about NAs
#TODO mappings of names

factors=['ad_group','campaign_id','day_week','device',
         'category_1st_level', 'category_2nd_level',
         'category_3rd_level', 'category_4th_level',
         'category_5th_level',
         'product_type_1st_level', 'product_type_2nd_level',
         'product_type_3rd_level', 'product_type_4th_level',
         'product_type_5th_level', 
         'brand', 'item_id',
         'category_1st_level*brand', 'category_2nd_level*brand', 'category_3rd_level*brand']

non_factors=[]
counts_events=pd.DataFrame({'Name':['CVR','CTR'],'counts':['clicks','impressions'],'events':['conversions','clicks']})
metrics_name=['cost']

sp = SparseCat(factors, non_factors, counts_events, metrics_names = metrics_name, alpha=0.75, lookup_df=lookup_df)

%time sp.fit(data_df)

# Wall time: 1min 10s

%time X = sp.transform(data_df)

# Wall time: 27min 26s

comment = [f + ':' + str(sp.factors_table_[f].shape[0]) for f in sp.factors]

comment = ','.join(comment) + '|' + ','.join(sp.non_factors)

%time scipy.io.mmwrite('data_201602_201603.mm', X, comment=comment, field='integer', precision=None, symmetry=None)

# manually reviewed groupings to check no duplicate subcategories... 
# only one is '--'
# set as missing?
cat1_month=calc_metrics(df_mg,'Category (1st level)',lookup_df = lookups_df)
yerr=np.full((cat1.shape[0],6),np.NaN)
yerr[:,2] = cat1['CTR%'] - cat1['CTR_0.12%']
yerr[:,3] = cat1['CTR_0.88%'] - cat1['CTR%']
yerr[:,4] = cat1['CVR%'] - cat1['CVR_0.12%']
yerr[:,5] = cat1['CVR_0.88%'] - cat1['CVR%']
fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True);
cat1.plot(x='Category (1st level)',kind='bar',y='Impressions', ax= axes[0])
cat1.plot(x='Category (1st level)',kind='bar',y='CTR%', yerr=yerr[:,2:4].T,ax= axes[1])
cat1.plot(x='Category (1st level)',kind='bar',y='CVR%', yerr=yerr[:,4:6].T, ax= axes[2])


cat1_month=calc_metrics(df_mg,['Category (1st level)','Month'],lookup_df = lookups_df)
fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True);
cat1_month.plot(x=['Category (1st level)','Month'],kind='bar',y='Impressions', ax= axes[0])
cat1_month.plot(x=['Category (1st level)','Month'],kind='bar',y='CTR%', ax= axes[1])
cat1_month.plot(x=['Category (1st level)','Month'],kind='bar',y='CVR%', ax= axes[2])


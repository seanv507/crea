# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 12:10:26 2014

@author: sean
"""
from __future__ import print_function
import pymysql as mysql
import pandas as pd
import pandas.io.sql as psql
import numpy as np
import sys


def MySQL_gettypes(con_dict, sql_table):
    host = con_dict['host']
    user = con_dict['user']
    passwd = con_dict['passwd']        
    db= con_dict['db']    
    db_info='information_schema'
    
    con = mysql.connect(host=host, user=user, passwd=passwd,
                          db=db_info)
    sql = 'select column_name, data_type from columns where '
    sql += "table_name='" + sql_table + "' and table_schema='" + db+"'"
    types_df = psql.frame_query(sql, con).set_index('column_name')
    con.close()

    return types_df


def fix_types(train_df, sql_types, int_null_value=0):
    # TODO do we want this or should we keep nans? and fix filesave
    # ideally we specify whether each field supports nulls in DB
    # and validate as it goes in/ and at DE
    # eg should banner size ever be Null?
    # whereas gender is predominantly null, 
    # and we should have separate coeffs for null, male,female
    # how to implement on ruby side? ie to lookup based on null 'value'
    # numpy doesn'y have int 'null' so pandas casts columns with Null into doubles
    # reconvert floats to ints replacing nan
    # data is replaced in place (to save memory)
    for col_name in train_df.columns:
        if sql_types.at[col_name,'data_type']=='int' and train_df.dtypes[col_name].name == 'float64':
            train_df[col_name].fillna(int_null_value, inplace=True)
            train_df[col_name] = train_df[col_name].astype(int)


#con_dict={'host','db','user','passwd'}
def MySQL_getfields(con_dict, df, sql_table, id_col, fields=None, verbose=False, rename_fields=None):
    ''' denormalise: return associated fields from a db table and merge into dataframe

               df: dataframe to merge into
        sql_table: table to query
           id_col: id column in data frame
           fields: list of columns in sql table to return- 
                   if empty return all fields
    '''
    host=con_dict['host']
    user=con_dict['user']
    passwd=con_dict['passwd']        
    db=con_dict['db']
    
    ids = df[id_col].dropna().astype(np.int).unique()
    con = mysql.connect(host=host, db=db,
                          user=user, passwd=passwd)
    if fields is None:
        fields_string = '*'
    else:
        fields_string = 'id, {fields} '.format(fields=','.join(fields))
    sql = 'select {fields} from {table} where id in ({ids})'.format(
        table=sql_table,
        fields=fields_string,
        ids=','.join(map(str, ids)))
    if verbose:
        print(sql)
    fields_df = psql.frame_query(sql, con).set_index('id')
    con.close()
    if rename_fields is not None:
        fields_df.rename(columns=dict(zip(fields, rename_fields)),
                         inplace=True)

    merge_df = pd.merge(df, fields_df,
                        left_on=id_col, right_index=True, how='left')
    return merge_df

def add_features( data_df):
    """ adds campaign_id, ad_account_id, pub_account_id ( as well as campaign_name and ad_name site_name)
    """
    con_dict_mad={'host':'db.lqm.io','db':'madvertise_production',
                  'user':'readonly','passwd':'z0q909TVZj'}
    data_df=MySQL_getfields(con_dict_mad, data_df,'ads','ad',['name','account_id','campaign_id'])
    # house_ads have no campaign_id, so will be NULL, which pandas converts campaignto floats and nans
    data_df.rename(columns={'name':'ad_name','account_id':'ad_account_id'},inplace=True)
    data_df=MySQL_getfields(con_dict_mad, data_df,'campaigns','campaign_id',
                            ['name'], rename_fields=['campaign_name'])
    data_df=MySQL_getfields(con_dict_mad, data_df,'sites','site',
                            ['name','account_id'], 
                            rename_fields=['site_name', 'pub_account_id'])
    return data_df


def MySQL_getdata(con_dict, sql_table, train_per, features, 
                  rtb_flag, where=None,sum_metrics=None,split_by_hours=None):
    # Getting data from MySql
    host=con_dict['host']
    user=con_dict['user']
    passwd=con_dict['passwd']        
    db=con_dict['db']
                  
    sql_types = MySQL_gettypes(con_dict, sql_table)
    sql_types.loc['hour']='int'
    sql_types.loc['dayofweek']='int'
    features=[f.lower() for f in features] # lower case for matching
    # screen out extra features that aren't in sql_table
    # aim is subsequent functions add those
    sql_table_features = set(features) & set(sql_types.index)
    select_features = ', '.join(sql_table_features)
    if sum_metrics is None:
        sum_metrics=['clicks', 'instances']
    sum_metrics=', '.join(['SUM(`{0}`) as {0}'.format(m) for m in sum_metrics]) 

    groupby_features = select_features[:]

    if 'hour' in sql_table_features:
        select_features= select_features.replace('hour', 'extract( hour from `date`) as hour')
    if 'dayofweek' in features:
        select_features= select_features.replace('dayofweek', 'dayofweek( `date`) as dayofweek')        
    if split_by_hours is not None:
        select_features="truncate(TIMESTAMPDIFF(hour, '{start}', date )".format(start=train_per[0]) \
            + '/{split_by_hours} ,0) as timeslot, '.format(split_by_hours=split_by_hours) \
            + select_features
        groupby_features = 'timeslot, ' + groupby_features

    where_clause=' WHERE date >= "{}"'.format(train_per[0]) \
             + ' AND date <= "{}"'.format(train_per[1]) \
             + ' AND rtb in ({})'.format(', '.join([str(f) for f in rtb_flag]))
  
    if where is not None:
        where_clause+=' AND (' + where +')'
    

    sql = 'SELECT ' + select_features + ',' + sum_metrics\
             + ' FROM `'+sql_table+'`'\
             + where_clause\
             +' GROUP BY ' + groupby_features
          
    print(sql, file=sys.stderr)
    con = mysql.connect(host=host, user=user, passwd=passwd, db=db)
    train_df = psql.frame_query(sql, con)
    con.close()

    fix_types(train_df, sql_types, -1)
    return train_df

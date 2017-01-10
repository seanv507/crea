# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 10:35:29 2014

@author: sean
"""

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import datetime
import mysql_lqm
import pymysql as mysql

def clean_site_names(sites_df,account_field,name_field,clean_name_field):
    # adds extra column in place
    adx=sites_df[account_field]==4774   
    sites_df[clean_name_field]=sites_df[name_field].str.replace('http://','')
    sites_df.loc[adx, clean_name_field]=sites_df.loc[adx, clean_name_field].str.replace('adx-app:','')
    sites_df.loc[adx, clean_name_field]=sites_df.loc[adx, clean_name_field].str.replace('adx-app-','')
    sites_df.loc[adx, clean_name_field]=sites_df.loc[adx, clean_name_field].str.replace('adx-site-','')
    sites_df.loc[adx, clean_name_field]=sites_df.loc[adx, clean_name_field].str.replace('adx-?-','')

    # should be identifying and dropping first colon...

    sites_df.loc[-adx, clean_name_field]=sites_df.loc[-adx, clean_name_field].str.extract('[^: ]:(.*)')
    

def campaigns_extend(campaigns):
    campaigns['days']=(campaigns['end_date'] \
        -campaigns['start_date'])/np.timedelta64(1,'D') # convert to days units
    campaigns['budget_euros'] = campaigns.budget_cents/100.0
    campaigns['costs_euros'] = campaigns.costs_cents/100.0
    campaigns['budget_euros_daily'] = campaigns.budget_euros/campaigns.days
    
    campaigns['ctr_lifetime%'] = campaigns.clicks_count/campaigns.impressions_count*100
    campaigns['eCPC_lifetime'] = campaigns.costs_euros/campaigns.clicks_count
    campaigns['eCPM_lifetime'] = campaigns.costs_euros/campaigns.impressions_count*1000
    isCPC = campaigns.budget_type == 'cpc'
    isCPM = campaigns.budget_type == 'cpm'
    campaigns['budget_clicks_daily']=campaigns.budget_clicks/campaigns.days
    campaigns['budget_ais_daily']=campaigns.budget_ais/campaigns.days
    campaigns.loc[isCPC,'budget_ais_daily']=campaigns.budget_clicks_daily[isCPC]/campaigns['ctr_lifetime%'][isCPC]*100

def read_campaigns(con_dict_mad, start_date,end_date):
    columns='id, name, account_id, created_at, updated_at, budget_type,'\
    + 'start_date, end_date, charge_cents, charge_currency, '\
    + 'budget_cents, budget_currency, daily_budget_cents, daily_budget_currency,'\
    + 'daily_budget_impressions, daily_budget_clicks,'\
    + 'click_count_today, impression_count_today, costs_today_cents,'\
    + 'clicks_count, impressions_count, costs_cents, budget_clicks, budget_ais,'\
    + 'capping_budget_cents, capping_budget_currency, capping_budget_clicks,'\
    + 'capping_budget_impressions, budget_downloads, daily_budget_downloads,'\
    + 'capping_budget_downloads, downloads_count, download_count_today, '\
    + 'budget_views, daily_budget_views, capping_budget_views, views_count, view_count_today,'\
    + 'rev_share, currency, budget_allocation_type,'\
    + 'enable_bid_optimization, min_bidfactor, max_bidfactor, earning_cents, earning_currency,'\
    + 'category_id, time_zone, state'

    sql='select ' + columns \
        + ' from campaigns where ' \
        + 'start_date<="{}"'.format(end_date.isoformat()) \
        + ' and end_date>="{}"'.format(start_date.isoformat())\
        + ' and state="active"'
    con_mad=mysql.connect(**con_dict_mad)
    print(sql)
    campaigns_df = psql.frame_query(sql, con_mad)
    con_mad.close()
    return campaigns_df
    

    

def get_campaign_data(con_dict_dse, start_date,end_date, where_str=None):
    """ read data from ctr_error, converting prices from microcents to euros"""
    where_str_ext='date>="{start}" and date<="{end}" and rtb!=0 {w}'.format(\
        start=start_date.isoformat(),end=end_date.isoformat(),w=where_str)
    con_dse=mysql.connect(**con_dict_dse)
    sql='''select date, campaign, ad, site, sum(bids) as bids, 
    sum(bid_price)/1e8 as bid_price, sum(bid_floor)/1e8 as bid_floor, 
    sum(ais) as ais, sum(win_bid_price)/1e8  as win_bid_price,
    sum(win_bid_floor)/1e8 as win_bid_floor, sum(clicks) as clicks,
    sum(e_clicks) as e_clicks, sum(downloads) as downloads, 
    sum(earnings)/1e8 as earnings from ctr_error where 
    {}
    group by date, campaign, ad, site'''.format(where_str_ext)
    print(sql)
    campaign_sites_df = psql.frame_query(sql, con_dse)
    con_dse.close()
    return campaign_sites_df


con_dict_mad={'host':'db.lqm.io','db':'madvertise_production',
                  'user':'readonly','passwd':'z0q909TVZj'}
con_dict_dse={'host':'db.lqm.io','db':'dse','user':'dse','passwd':'dSe@lQm'}

campaigns=read_campaigns(con_dict_mad,
               start_date=datetime.date(year=2014,month=12,day=12),
               end_date=datetime.date(year=2014,month=12,day=12))
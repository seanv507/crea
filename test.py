# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 17:21:22 2014

@author: sean
"""


import datetime
import os

DATA_DIR = '.'
NOW = datetime.datetime(year=2014,month=10,day=24,hour=0,minute=0)
INTERVAL = 14  # 2 weeks time interval
MODEL_NAME = 'logistic_regression'


params = ' '.join([
  '--features', 'site+ad+site*ad+banner_width*banner_height+site*banner_width*banner_height+device+site*device',
  '--start', '"'+(NOW-datetime.timedelta(days=INTERVAL)).strftime("%Y-%m-%d %H:%M:%S")+'"',
  '--end', '"' +NOW.strftime("%Y-%m-%d %H:%M:%S")+'"',
  '--bias', '1',
  '--tolerance', '0.00000001',
  '--regularization', '0.37170075843439998',
  '--data_dir', DATA_DIR,
  '--mysql_host', 'db.lqm.io',
  '--mysql_name', 'dse',
  '--mysql_user', 'dse',
  '--mysql_pass', 'dSe@lQm' 
 ])

cmd = "python ../src/libLinear_MySQL.py "+params
print cmd
fail=os.system(cmd)
if fail:
    print 'error running '+cmd



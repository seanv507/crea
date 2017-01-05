# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:45:52 2016

@author: Sean Violante
"""

import pandas as pd


#shopping = pd.read_csv('shopping-performance-report.csv', encoding='utf-8')
#shopping.to_excel('shopping-performance-report.xlsx', encoding='utf-8')

all_reports= pd.read_csv('all-reports.csv', encoding='utf-8')

all_reports.to_excel('all-reports.xlsx', encoding='utf-8')
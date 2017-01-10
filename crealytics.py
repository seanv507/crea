#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:14:13 2017

@author: sean
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from scipy.stats import beta
import matplotlib.pyplot as plt
import sqlalchemy 

def do_lookup(cat, typ):
    lookups[cat]={}
    
    def lookup( val):
        if val not in lookups[cat]:
            lookups[cat][val]=typ(len(lookups[cat]))
        return lookups[cat][val]
    return lookup
    
def map_lookup(df, lookup_df=None, subset=None):
    if lookup_df is None:
        return
    cats = set(df.columns) & set(lookup_df.index.levels[0])
    if subset:
        cats &= subset 
    for c in cats:
       df[c] = df[c].map(lookup_df.xs(c).id) 

#unnecessary? because can pass category as dtype in read_csv (new pandas version?)
def cast_categ(df_inp, cat_types):
    for c in cat_types:
        df_inp[c]=df_inp[c].astype('category')
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:50:18 2014

@author: sean
"""

import sklearn.linear_model.logistic as logi

clf=logi.LogisticRegression(tol = 0.00000001, C=0.372)

clf.fit(sparse_train,targets,weights)

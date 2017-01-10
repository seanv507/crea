# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 23:16:10 2014

@author: sean
"""
import redis
import numpy as np
import re
import json

def logist(x):
    """
        Computes logist(-x) ie opposite sign to normal logistic function.
        http://en.wikipedia.org/wiki/Logistic_function
        the sign diff is because liblinear code takes first class in input file
        as class to predict ( and we input no-click data first )
    """
    return 1/(1+np.exp(x))

class LogisticRegression:
    MISSING_VALUE=-1
    # id value for missing feature
    def __init__(self, features,constant, coeffs):
        self.features = features[:]
        self.constant = constant
        self.coeffs = coeffs.copy()
        
    def get_features(self, inpt):
        """ 
        inpt - dictionary of feature ids eg {'site':10090, 'ad': 3456}
        returns array of LR inputs ( mssing items are ignored)
        #TODO decide representation of 'missing' empty fields  
        """
        # 
        
        return [ 
            ','.join(map(lambda dimn: dimn+':'+str(inpt.get(dimn,self.MISSING_VALUE)),feat))\
                    for feat in self.features
                        if all([f in inpt for f in feat]) \
                           ]
                    
    def value(self, inpt):
        """ return probability prediction
        inpt - dictionary of feature ids eg {'site':10090, 'ad': 3456}
        (assumes categorical variables) 
        - so input is categories eg site id ...
        - any missing categories will be ignored( zero value) 
        """
        features = self.get_features(inpt)
        vals = [self.coeffs.get(feature, 0) for feature in features]
        features.insert(0,'constant')
        vals.insert(0,self.constant)
        return logist(sum(vals)), zip(features,vals)
    
    def analyse(self, value_output):
        """Returns sorted dataframe of features and corresponding probability output.

        Args:
            value_output: output of value function
        Returns:
            Pandas dataframe of features sorted by absolute value
                (ie in descending order of importance)
                logist% column is probability as percentage
                for cumulative sum of features
        """
        import pandas as pd
        df=pd.DataFrame.from_records(value_output[1],
                                  index='feature',
                                  columns=['feature','value'])
        df['abs']=df.value.abs()
        df.sort('abs',ascending=False,inplace=True)
        df['cumval']=df.value.cumsum()
        df['logist%']=logist(df.cumval)*100
        return df

    @classmethod
    def from_redis(cls, redisdb, key):
        coeffs = redisdb.hgetall(key)
        features = json.loads(coeffs['Features'])
        constant = float(coeffs['Constant'])
        return cls(features, constant, coeffs)
    
    @classmethod
    def from_file(cls, filename):
        with open(filename) as fp:
            # "factors" = ('site', 'ad', 'site''*ad')            
            li = fp.readline() 
            b1 = li[13:-2].replace("'","").split(', ')
            features = [b.split('*') for b in b1]
            non_factors = fp.readline()
            li = fp.readline()
            m = re.search('"factors length" = \[([^\]]+)\]',li)

            factors_length = map(int,m.group(1).split(', '))
            li=fp.readline()
            m = re.search('"bias" = (.+)',li)
            constant = float(m.group(1))
            
            # feature : value
            # feature,  
            #   "ad : 36862" = -0.001043236431905328
            #   "site*ad : (41353, 36064)" = -9.482542017851045e-05
            # 
            # find "word : 0/1( not " or ) 0/1 ) " 
            ma = re.findall('"([^ ]+) : \(?([^)"]+)\)?" = (.+)$',
                            fp.read(), re.MULTILINE)
            # group[0]=feature eg site*ad
            # group[1]=feature id eg 123,154
            
            coeffs = {','.join(
                map(lambda x :x[0]+':'+x[1],
                    zip(m[0].split('*'),m[1].split(', '))))
                : float(m[2])    for m in ma}

#            if c = line.match('"([^ ]+) : \(?([^)"]+)\)?" = (.+)$')
#                attributes = c[1].split(/\*\s*/)
#                ids = c[2].split(/,\s*/)
#                coef = c[3].to_f
#                key = attributes.zip(ids).map{|attr,id| attr + ':' + id}.join(',')
#                redis.hset(model_name, key, coef)            
            return cls(features,constant,coeffs)
    
    

#redisNow = redis.StrictRedis(host='db.lqm.io', port=6379, db=0)
#
#ks=redisNow.keys()
#ks
#coefs=redisNow.hgetall('logistic_regression')
#features=coefs['Features']
#print features


if __name__ == "__main__":
    lr1=LogisticRegression.from_file('lookup.txt')
    d={'campaign_id':5963, 'ad':5436065,'banner_height':960,'banner_width':640,'device':263}
    print lr1.get_features(d)
    print lr1.value(d)
    

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 01 11:03:40 2013

@author: sv507
"""
from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import collections
from scipy.stats import beta

def map_lookup(df, lookup_df=None, subset=None):
    # assumes lookup_df has index ['id_name','id_index'])
    if lookup_df is None:
        return
    cats = set(df.columns) & set(lookup_df.index.levels[0])
    if subset:
        cats &= subset 
    for c in cats:
       df[c] = df[c].map(lookup_df.xs(c).id) 


def save_sparse(filename,matrix):
    np.savez(filename,data=matrix.data,indices=matrix.indices,indptr=matrix.indptr)

def csr_write_libsvm(filename,X_csr, y, nfeatures):
    """ write to libsvm format
    the CSR matrix is assumed to be 'regular' nrows x nfeatures
    ( so eg a feature with zero value should also be included)
    """
    #assert X_csr.shape[0]==y.shape[0]
    #assert X_csr.shape[0]*nfeatures==X_csr.indices.shape[0]
    with open(filename,'wb') as fp:
        for d in zip(y,X_csr.indices.reshape(-1,nfeatures),
                     X_csr.data.reshape(-1,nfeatures)):
            print >>fp,'{}'.format(d[0]),
            for d1 in zip(d[1]+1,d[2]) :
                if d1[1]: print >>fp,'{:.0f}:{:.0f}'.format(*d1),
            print >>fp
        # add 1 to d[1] because libsvm requires 1 based index

def csr_subrows(X_csr,indices, nfeatures):
    """ ? not sure if necessary supposed to be for removing columns
        the CSR matrix is assumed to be 'regular' nrows x nfeatures
    ( so eg a feature with zero value should also be included)
    """

    return csr_matrix((
        X_csr.data.reshape(-1,nfeatures)[indices,:].ravel(),
        X_csr.indices.reshape(-1,nfeatures)[indices,:].ravel(),
        nfeatures*np.arange(len(indices)+1)))
            
# '['Impressions','Clicks','Conversions', 'Cross-device conv.', 'Total conv. value']

def calc_metrics(df,gps,counts_events,metrics_names,alpha=0.75, lookup_df = None):
    # http://stackoverflow.com/a/922796/
    metrics = counts_events.counts.tolist() 
    metrics += [c for c in counts_events.events.tolist() if c not in metrics] 
    metrics += metrics_names
    met=df.groupby(gps)[metrics].sum()
#    met.reset_index(inplace=True)
#    # map on index doesn't accept series
    if isinstance(gps, basestring):
        gps=[gps]
    for i_g, g in enumerate(gps):
        if g in lookup_df.index.levels[0]:
        
            #met[g + '_name'] = met.index.map(lookup_df.xs(g).id)
            # taken from series.map source            
            # indexer = met.index.get_indexer(lookup_df.xs(g).id)
            if met.index.ndim==1:
                met[g + '_name'] = met.index.to_series().map(lookup_df.xs(g).id)
            else:
                met[g + '_name'] = met.index.level[i_g].to_series().map(lookup_df.xs(g).id)
            #TODO either keep as separate dfs in ordered dict or need to turn into 1 d index/name    
            #TODO  clumsy
            
    # but sparsecat expects mapping as index
    for e in counts_events.itertuples():
        met[e.Name + '%']=met[e.events].div(met[e.counts])*100
        x = beta.interval(alpha, met[e.events]+.5, met[e.counts]-met[e.events]+.5)
        met[e.Name + '_{:.2f}%'.format(.5 - alpha/2.0)] = x[0]*100
        met[e.Name + '_{:.2f}%'.format(.5 + alpha/2.0)] = x[1]*100
    
    return met
  
# take id metrics (eg clicks) etc
# so sum all whichever variable we are 'counting'

# problem 
# we want factor table both for univariate analysis and for data processing!!!

class SparseCat:
    def __init__(self, factors, non_factors, 
				counts_events, metrics_names = None, alpha=0.75, lookup_df=None ):
        # a list of named features ("site",...,'site*ad')
        # make copy
        self.factors=tuple(factors)
        self.non_factors=tuple(non_factors)
        self.counts_events = counts_events.copy() #dataframe...
        self.metrics_names = metrics_names
        self.alpha = alpha
        self.lookup_df = lookup_df
        
    #def get_params():
    #def set_params(**params):
        
    def fit(self,X,y = None):
        # don't use y
        # take pandas array and convert to csr representation
        
        factors_tables = []
        self.len_factors = []
        for f in self.factors:
            fs=f.split('*') # either 'ad' or 'ad*site*device' etc
            # keep ordering or not? no checking of duplicates etc
            # why use y at all? X has to contain clicks anyway
            table = calc_metrics(X, fs, 
                             self.counts_events, self.metrics_names, 
                             self.alpha, self.lookup_df)
            table.sort_values(self.counts_events.loc[0,'counts'], 
                              inplace = True, 
                              ascending = False)
            if len(fs)>1:
                table.index = table.index.map(lambda x:'*'.join(map(str,x)))
            # to concatenate we need to have the same index dimensions!! making lookup harder?
            factors_tables.append(table)
            self.len_factors.append(table.shape[0])
                
        self.factors_table = pd.concat(factors_tables, keys=self.factors)
        self.factors_table['col_index']=np.arange(self.factors_table.shape[0])
       
        #TODO identify NA !!!
        

        self.nfeatures_ =len(self.factors)+len(self.non_factors) #
        self.start_non_factor_ = sum(self.len_factors)
        
        self.ncols_ = self.start_non_factor_ + len(self.non_factors) 
        # dummy variables + other

        # should handle lim cases of zero factor/ non factor
        self.start_features_=np.concatenate(([0],
                                       np.cumsum(self.len_factors[:-1]),
                                np.arange(self.start_non_factor_,
                                          self.start_non_factor_ \
                                          + len(self.non_factors))))
        self.columns = self.factors_table.index.levels[1].tolist() + list(self.non_factors)
        
    
    def set_params(self,count_cutoff):
        self.count_cutoff=count_cutoff
    
              
    def transform(self,X):
        ndata=X.shape[0] # original data length
        # idea is that we set "irrel" to zero and all other
        
        indices=np.zeros((ndata, self.nfeatures_),dtype=np.int)
        vals=np.ones((ndata,self.nfeatures_))
        # then overwrite ones for Non mapped and non_factors
        for ifactor,factor in enumerate(self.factors):
            factor_split=factor.split('*')
            
            # mult or single !!!
            if len(factor_split)==1:
                keyer = X[factor]
            else:
                keyer = X[factor_split].apply(tuple, axis=1)
            
            index = keyer.map( self.factor_tables['col_index'].xs(factor))
            #was index=X[factor_split].apply(lambda x: self.mappings_[factor][x],axis=1)
            #but see http://stackoverflow.com/q/22293683
            
            # identify vals failed to map
            vals[index.isna()] = 0
            index = index.fillna(self.start_features_[ifactor])

        for i,non_factor in enumerate(self.non_factors):
            indices[:, len(self.factors) + i] = self.start_non_factor_ + i
            vals[   :, len(self.factors) + i] = X[non_factor]
        
        X_sparse=csr_matrix((vals.ravel(),indices.ravel(),self.nfeatures_ \
            * np.arange(ndata+1)))
        X_sparse.has_sorted_indices = True        
        return X_sparse


#if __name__ == "__main__":
#    
#    
#    
#    sc_new=SparseCat(factors,non_factors)
#    sc_new.fit(small,small['clicks'])
#    sc_new.set_params(count_cutoff=10)
#    z_new=sc_new.transform(small)
import pandas as pd
import crealytics


def calc_metrics_test():
	df = pd.DataFrame({'a':['a','b','a'], 'c':['c','d','d'],'clicks': [0,1,1],'impressions':[1,10,1], 'cost': [100,200,300]})
	gps = ['a','c']
	counts_events =pd.DataFrame({'Name': ['CTR'],'counts':['impressions'],'events': ['clicks']})
	metrics_names=['cost']
	calc_metrics(df,gps,counts_events,metrics_names,alpha=0.75, lookup_df = None)
     met_check = met.read_pickle('test_calc_metric.pkl')
	assert met.equals(met_check)

def make_factor_table_weighted_test():
    te = pd.DataFrame({'a':[0,1,0],
				  'clicks':[0,0,1],
				       'b':[1,2,3]})
    wts=np.array([1,3,5]).reshape(3,)
    d1, c1 = make_factor_table_weighted(te, 'a', wts)


df = pd.DataFrame({'a':['a','b','a'], 'c':['c','d','d'],'clicks': [0,1,1],'impressions':[1,10,1], 'cost': [100,200,300]})
gps = ['a','c']
counts_events =pd.DataFrame({'Name': ['CTR'],'counts':['impressions'],'events': ['clicks']})
metrics_names=['cost']
met = calc_metrics(df,gps,counts_events,metrics_names,alpha=0.75, lookup_df = None)
	
	
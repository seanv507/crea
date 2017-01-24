require(arules)

df=data.frame(aa=c('a','d','d','d'), bb=c('l','m','l'),cc=c('p','p','p','p'))
trans = as(df,'transactions')
trans_freq <-apriori(trans, parameter = list(support= 0.1, target="frequent"))

inspect(trans_freq)

summary(trans_freq)

# what does most frequent item sets report?

# most frequent items:
#     bb=l    cc=p    aa=a    aa=b    aa=d (Other) 
#        2       2       0       0       0       0 
# 

# shouldn't most frequent itemset be {cc=p} - is it using minlen 2?
# inspect(trans_freq)
# items       support  
# [1] {bb=l}      0.6666667
# [2] {cc=p}      1.0000000
# [3] {bb=l,cc=p} 0.6666667


# what does this mean? that there are 1534202 combinations of length 6? that have support   6e-04 = >1000 occurrences
# element (itemset/transaction) length distribution:sizes
# 2       3       4       5       6 
# 22274  133794  456277 1005942 1534202 


# support 10000 occurrences
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 2.000   5.000   6.000   5.564   7.000  10.000 

# this means 50% of the frequent itemsets are of size 6 ..

#find itemsets with 2 or more items
inspect(subset(trans_freq, subset=size(items)>1))



data(Adult)
summary(Adult)
rules = apriori(Adult,parameter=list(support=0.2,confidence=0.8))
summary(rules)

data(Groceries)
groc_freq <- apriori(Groceries, parameter = list(support=0.005, target="frequent", minlen=1, maxtime=30))
Groceries_multilevel <- addAggregate(Groceries, "level2")

groc_freq <- apriori(Groceries, parameter = list(support=0.005, target="frequent", minlen=1, maxtime=30))
groc_multi_freq <- apriori(Groceries_multilevel, parameter = list(support=0.005, target="frequent", minlen=1, maxtime=30))
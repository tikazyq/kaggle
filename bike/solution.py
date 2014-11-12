# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Inspired by [an article from Brandon Harris link](http://brandonharris.io/kaggle-bike-sharing/) using **party**, an R package using conditional inference trees, which relies on calculation of covariates statistics. This seems making sense, because some variabels are correlated, like *season*, *temp*, *weather*.
# 
# Kaggle evaluation function: $$\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$$

# <markdowncell>

# #### Prepare dataset

# <codecell>

from sklearn import linear_model as lm
from sklearn import preprocessing as pp
import statsmodels.api as sm
from pandas import *
import pandas as pd
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 8)

np.random.seed(1000)
cat_vars = [
            'season',
            'holiday',
            'workingday',
            'weather',
            'weekday',
            'hour'
            ]
num_vars = [
            'temp',
            'atemp',
            'humidity',
            'windspeed'
            ]
out_vars = [
            'casual',
#             'registered',
#             'count'
            ]

# fit_vars = cat_vars + num_vars

def load_dataset(fname):
    d = read_csv(fname, parse_dates=[0])
    d['hour'] = d['datetime'].apply(lambda x:x.hour)
    d['weekday'] = d['datetime'].apply(lambda x:x.weekday())
    for var in cat_vars:
        d[var] = d[var].astype(str)
    d_cat = pd.get_dummies(d.ix[:,cat_vars])
    d = d.join(d_cat)
    
    cat_vars2 = d_cat.columns.tolist()
    global fit_vars
    fit_vars = cat_vars2 + num_vars
    return d
    
def split_dataset(d, ratio=0.7):
    d['randn'] = np.random.uniform(size=len(d))
    return (d[d['randn']<ratio].drop('randn',axis=1), 
            d[d['randn']>=ratio].drop('randn',axis=1))

def rmsle(y, y_p):
    return (1.0/len(y) *sum((log(y+1) - log(y_p+1))**2))**0.5

def conv_cat_vars(d, var):
    if var in cat_vars:
        return pd.get_dummies(d.ix[:,[var]])
    return d.ix[:,[var]]

train = load_dataset('train.csv')
d_train, d_cal = split_dataset(train)
test = load_dataset('test.csv')

# <markdowncell>

# #### Ridge Regression using sklearn

# <codecell>

coeffs = []
rmsles = []
alphas = logspace(-2, 10, 100)
min_rmsle_cal = 1e9
min_rmsle_train = 1e9
best_alpha = 1e9
for alpha in alphas:
    x = d_train.ix[:,fit_vars]
    y = array(d_train.ix[:,out_vars])
    clf = lm.Ridge(alpha)
    clf.fit(x, y)
    coeffs.append(clf.coef_[0])
    
    x_c = d_cal.ix[:,fit_vars]
    y_c = array(d_cal.ix[:,out_vars])
    y_cp = clf.predict(x_c)
    y_cp = y_cp * (y_cp>=0)
    y_p = clf.predict(x)
    y_p = y_p * (y_p>=0)
    current_rmsle_cal = rmsle(y_c, y_cp)
    current_rmsle_train = rmsle(y, y_p)
    rmsles.append(current_rmsle)
    if current_rmsle_cal < min_rmsle_cal:
        min_rmsle_cal = current_rmsle_cal
        min_rmsle_train = current_rmsle_train
        best_alpha = alpha
    global test
    y_t = test.ix[:,out_vars]
#     DataFrame(hstack((y_t, y_c)))
print "min rmsle (cal): %s" % min_rmsle_cal
print "min rmsle (train): %s" % min_rmsle_train
print "best alpha: %f" % alpha
ax = plt.gca()
plt.plot(alphas, array(coeffs))
ax.set_xscale('log')
plt.legend(fit_vars, 7)
plt.plot(alphas, rmsles)
plt.title('Ridge Regression (coeffs)')
plt.show()

# <markdowncell>

# #### Single variable evaluation

# <codecell>

fit_vars = cat_vars + num_vars
coeffs = []
for var in fit_vars:
    clf = lm.LinearRegression()
    y = train.ix[:,out_vars]
    x = conv_cat_vars(train, var)
    dt,dc = split_dataset(conv_cat_vars(train, var))
    xt,yt = train.ix[dt.index,x.columns].fillna(0), train.ix[dt.index,out_vars].fillna(0)
    xc,yc = train.ix[dc.index,x.columns].fillna(0), train.ix[dc.index,out_vars].fillna(0)
    clf.fit(xt, yt)
    yc_p = clf.predict(xc)
    print "RMSLE for %s: %f" % (var, rmsle(yc_p, yc))
    coeffs.append((var, clf.coef_, rmsle(yc_p, yc)))
                  
# hour is the most important feature
var = 'hour'
clf.coef_ = filter(lambda x: x[0]==var, coeffs)[0][1]
x = array(conv_cat_vars(test, var))
clf.predict(x)



# Kaggle evaluation function: $$\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$$

# #### Prepare dataset

# In[15]:

from sklearn import linear_model as lm
from sklearn import preprocessing as pp
import statsmodels.api as sm
from pandas import *
import pandas as pd
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
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
#             'casual',
#             'registered',
            'count'
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
    return d
    
def split_dataset(d, ratio=0.9):
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


# #### Decision Tree Regression with sklearn

# In[16]:

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

fit_vars = cat_vars + num_vars
for var in cat_vars:
    train[var] = train[var].astype('str')

Xt, yt = d_train.ix[:,fit_vars], d_train.ix[:,out_vars]
Xc, yc = d_cal.ix[:,fit_vars], d_cal.ix[:,out_vars]

perf = []
best_fi = []
best_ec = Inf
for msl in range(1,21,5):
    for mss in range(2, 40,1):
        # comment the below two lines out to choose regressor
        clf = DecisionTreeRegressor(min_samples_split=mss,min_samples_leaf=msl,splitter='best')
#         clf = RandomForestRegressor(n_estimators=10, min_samples_split=mss,min_samples_leaf=msl)
    
        clf.fit(Xt, yt)
        yt_p = mat(clf.predict(Xt)).T
        yc_p = mat(clf.predict(Xc)).T
        
        et = rmsle(array(yt),array(yt_p))
        ec = rmsle(array(yc),array(yc_p))
        
        if ec < best_ec:
            best_ec = ec
            best_fi = clf.feature_importances_
        perf.append((msl, mss, et, ec))

perf = DataFrame(perf, columns=['msl','mss','et','ec'])
print perf.sort('ec',ascending=1).head()

ax1 = plt.subplot(211)
ax1.set_title('training data rmsle')
ax2 = plt.subplot(212)
ax2.set_title('calibration data rmsle')
perf.pivot('mss','msl', 'et').plot(ax=ax1)
perf.pivot('mss','msl', 'ec').plot(ax=ax2)
plt.tight_layout()
plt.show()

p = DataFrame(array([fit_vars, best_fi]).T,columns=['feature','importance']).sort('importance',ascending=0)
print p


# Out[16]:

#         msl  mss        et        ec
#     34    1   36  0.375149  0.448858
#     33    1   35  0.373483  0.450221
#     36    1   38  0.378482  0.450499
#     35    1   37  0.376975  0.450678
#     32    1   34  0.371426  0.451113
# 

# image file:

#           feature  importance
#     5        hour  0.67858956
#     6        temp  0.26677283
#     0      season  0.02342060
#     4     weekday  0.01751396
#     7       atemp  0.00406671
#     3     weather  0.00339767
#     2  workingday  0.00292615
#     8    humidity  0.00218973
#     9   windspeed  0.00061127
#     1     holiday  0.00051148
# 

# In[17]:

# DecisionTreeRegressor
# Choose the optimised params (min_samples_leaf=6, min_samples_split=37)
clf = DecisionTreeRegressor(min_samples_leaf=6, min_samples_split=37)
clf.fit(train.ix[:,fit_vars],train.ix[:,out_vars])
submit = test.ix[:,['datetime']]
submit['count'] = clf.predict(test.ix[:,fit_vars])
submit.to_csv('submit_sklearn_dc.csv',index=0)

# RandomForesetRegressor
# Choose the optimised params (min_samples_split=20)
clf = RandomForestRegressor(n_estimators=10, min_samples_split=1,min_samples_leaf=20)
clf.fit(train.ix[:,fit_vars],train.ix[:,out_vars])
submit = test.ix[:,['datetime']]
submit['count'] = clf.predict(test.ix[:,fit_vars])
submit.to_csv('submit_sklearn_rf.csv',index=0)


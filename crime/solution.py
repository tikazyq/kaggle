from pandas import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize, LabelBinarizer


# helpers
def combine(df, lb):
    if hasattr(lb, 'classes_'):
        T = lb.transform(df.Category)  # target
    else:
        T = lb.fit_transform(df.Category)

    df = df.sort(['Dates', 'Category'])
    df.index = range(len(df))
    key_name = [
        'Dates',
        'PdDistrict',
        'Address',
        'X',
        'Y'
    ]

    F_ret = []
    T_ret = []

    for i in df.index:

        if i == 0:
            continue

        key = df.ix[i, key_name].tolist()
        key_prev = df.ix[i - 1, key_name].tolist()

        target = T[i]
        target_prev = T[i - 1]

        F_ret.append(key)
        if key == key_prev:
            T_ret.append(np.array([target, target_prev]).sum(axis=0).tolist())
        else:
            T_ret.append(target)

        if i > 0 and not (i % 5000):
            print i

    return F_ret, T_ret, lb


# load data
train_ = read_csv('train.csv', parse_dates=[0])

# feature engineering
train_['HourOfDay'] = train_['Dates'].apply(lambda x: x.hour)

# binarize target variables
lb = LabelBinarizer()
F_train, T_train, lb = combine(train_, lb)  # transform

# train test data split
# train, cal = train_test_split(train_, train_size=0.3, random_state=0)
test = read_csv('test.csv')
F_test, T_test, lb = combine(test, lb)

features = [
    'HourOfDay'
    'PdDistrict',
    'Address',
    'X',
    'Y'
]
labels = [
    'Category'
]

# train classifier

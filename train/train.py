import os
import gc
import glob
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from itertools import groupby
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor


de_train = pd.read_parquet('../data/de_train.parquet')
id_map = pd.read_csv('../data/id_map.csv')

sample_submission = pd.read_csv('../data/sample_submission.csv', index_col='id')


xlist = ['cell_type', 'sm_name']
_ylist = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']

y = de_train.drop(columns=_ylist)

train = pd.get_dummies(de_train[xlist], columns=xlist)

test = pd.get_dummies(id_map[xlist], columns=xlist)

uncommon = [f for f in train if f not in test]

X = train.drop(columns=uncommon)


model = LinearSVR(max_iter=2000, epsilon=0.1)

wrapper = MultiOutputRegressor(model)
wrapper.fit(X, y)

submission1 = pd.DataFrame(wrapper.predict(test), columns=de_train.columns[5:])
submission1.index.name = 'id'
submission1.to_csv('submission1.csv')


y1 = y.iloc[:, :1000].copy()
y2 = y.iloc[:, 1000:2000].copy()
y3 = y.iloc[:, 2000:3000].copy()


X1 = X.copy()
X2 = X1.join(y1)
X3 = X2.join(y2)

test1 = test.copy()

model = LinearSVR()
wrapper = MultiOutputRegressor(model)

wrapper.fit(X1, y1)

pr1 = pd.DataFrame(wrapper.predict(test1), columns= de_train.columns[5:1005])

test2 = test1.join(pr1)

wrapper.fit(X2, y2)

pr2 = pd.DataFrame(wrapper.predict(test2), columns= de_train.columns[1005:2005])

test3 = test2.join(pr2)

wrapper.fit(X3, y3)

pr3 = pd.DataFrame(wrapper.predict(test3), columns= de_train.columns[2005:3005])

test4 = test3.join(pr3)

t = test4.iloc[:, 131:].copy()

submission = t.join(submission1.iloc[:, 3000:])
submission.index.name = 'id'

submission.to_csv('submission.csv')

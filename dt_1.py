import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as mt, tree
train_data = pd.read_csv('C:\Users\Ashish\Documents\msc\ml_2016\CE802_Ass_2016\ce802ass_train.csv')
train_x = train_data.iloc[:850,:8]
train_y = train_data.iloc[:850,-1]
test_x = train_data.iloc[850:,:8]
test_y = train_data.iloc[850:,-1]
K_s = np.arange(1,11)

dt = DecisionTreeClassifier(random_state=0)
clf = GridSearchCV(estimator=dt, param_grid=dict(min_samples_leaf=K_s), n_jobs=-1)
clf.fit(train_x, train_y)
#svc.degree = 2
scores = cross_val_score(dt, train_x, train_y, cv=10)
pred_y = clf.predict(test_x)
cohen = mt.cohen_kappa_score(pred_y, test_y)
accu = mt.accuracy_score(pred_y, test_y)
confusion = mt.confusion_matrix(pred_y, test_y)
print("Cohen score:= {}. \nAccuracy score = {}. \nConfusion matrix = {}".format(cohen, accu, confusion))
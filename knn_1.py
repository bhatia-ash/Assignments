import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
train_data = pd.read_csv('C:\Users\Ashish\Documents\msc\ml_2016\CE802_Ass_2016\ce802ass_train.csv')
train_x = train_data.iloc[:,:8]
train_y = train_data.iloc[:,-1]
#C_s = np.logspace(-2, 2, 10)
K_s = np.arange(1,11)
knn = KNeighborsClassifier()
clf = GridSearchCV(estimator=knn, param_grid=dict(n_neighbors=K_s), n_jobs=-1)
clf.fit(train_x, train_y)
#svc.degree = 2
scores = cross_val_score(clf, train_x, train_y)

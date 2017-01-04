print(__doc__)


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm
train_data = pd.read_csv('C:\Users\Ashish\Documents\msc\ml_2016\CE802_Ass_2016\ce802ass_train.csv')
train_x = train_data.iloc[:,:8]
train_y = train_data.iloc[:,-1]
C_s = np.logspace(-2, 2, 10)
svc = svm.SVC(kernel='rbf')
clf = GridSearchCV(estimator=svc, param_grid=dict(C=C_s), n_jobs=-1)
clf.fit(train_x, train_y)
#svc.degree = 2
scores = cross_val_score(clf, train_x, train_y)

#scores = list()
#scores_std = list()
#for C in C_s:
#    svc.C = C
#    this_scores = cross_val_score(svc, train_x, train_y, n_jobs=1)
#    scores.append(np.mean(this_scores))
#    scores_std.append(np.std(this_scores))

# Do the plotting
#import matplotlib.pyplot as plt
#plt.figure(1, figsize=(4, 3))
#plt.clf()
#plt.semilogx(C_s, scores)
#plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
#plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
#locs, labels = plt.yticks()
#plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
#plt.ylabel('CV score')
#plt.xlabel('Parameter C')
#plt.ylim(0, 1.1)
#plt.show()
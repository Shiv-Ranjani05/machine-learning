from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import make_classification

X,y = make_classification(
    n_samples=1500, n_features=30, n_informative=20, n_classes=4, random_state=48
)

c_space = np.logspace(-8,10,20)
param_grid =  {'C':c_space}

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg, param_grid, cv=8 )
logreg_cv.fit(X,y)
print("Tuned logistic regression parameters:{}".format(logreg_cv.best_params_))
print("BEst score is {}".format(logreg_cv.best_score_))
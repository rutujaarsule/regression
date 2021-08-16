import pandas as pd
import numpy as np

df = pd.read_csv("K:\Advance Ana Portal Notes\ML lab/Concrete_Data.csv")
dum_df = pd.get_dummies(df, drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import ElasticNet

X = dum_df.iloc[:,0:-1]
y = df.iloc[:,-1]

# Create training and test sets

df = pd.read_csv("K:\Advance Ana Portal Notes\ML lab\Kfold/insurance.csv")
dum_df = pd.get_dummies(df)

X = dum_df.drop('charges',axis=1)
y = dum_df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                    random_state=42)

clf = ElasticNet(alpha=2, l1_ratio=0.6)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

######################################################################################

parameters = dict(alpha=np.linspace(0.001,40),
                  l1_ratio=np.linspace(0.001,1,10))
from sklearn.model_selection import GridSearchCV
clf = ElasticNet()
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2021,shuffle=True)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)

# Viewing CV Results
df_cv = pd.DataFrame(cv.cv_results_)

# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

# Object of best model
best_model = cv.best_estimator_

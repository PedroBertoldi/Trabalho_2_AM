import Data
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
import numpy as np

print("Getting Data")
x_array,y_array,enum = Data.GetData(cutof_percentage=0.0,n_lines=150000)

rfe = RFE(estimator=DecisionTreeClassifier())
new_x_array = rfe.fit_transform(x_array,y_array)
print("trainig")
x_train, x_test,y_train,y_test = Data.SplitTestAndTrain(new_x_array,y_array)

regr = BaggingRegressor(base_estimator=SVR(C=1.0, epsilon=0.2,verbose=True),n_jobs=-1,verbose=True).fit(x_train, y_train)
result = regr.score(x_test,y_test)
print("R: ",result)

# regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# regr.fit(x_train,y_train)
# print(regr.score(x_test,y_test))
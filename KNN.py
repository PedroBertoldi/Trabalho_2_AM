from numpy.core.fromnumeric import mean
import Data_bikes as Data
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import sys
import warnings
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

x_array,y_array,enum = Data.GetData(n_lines=None)

rfe = RFE(estimator=DecisionTreeClassifier())
x_array = rfe.fit_transform(x_array,y_array)

#Parametros =====================================================
hiperparametros =  [{"n_neighbors" : 2, "ID":"set_1"},
                    {"n_neighbors" : 20, "ID":"set_2"},
                    {"n_neighbors" : 200, "ID":"set_3"},
                    {"n_neighbors" : 500, "ID":"set_4"},]

dicionario = dict(n_neighbors=randint(0,500))

sufixo = "KNN"

if dicionario != None:
    train_model = KNeighborsRegressor()
    modelo = RandomizedSearchCV(train_model,dicionario,n_iter=10000,n_jobs=-1)
    search = modelo.fit(x_array,y_array)
    hiperparametros.append({"n_neighbors":search.best_params_["n_neighbors"], "ID" : "RandonSerachBest"})
#Fim Parametros =================================================

if not os.path.exists(sufixo):
    os.makedirs(sufixo)

kf = KFold(n_splits=5,shuffle=True)
for item in hiperparametros:
    best = [None,None,None]
    r2 = []
    mse = []
    for train_index, test_index in kf.split(x_array,y_array):
        x_train = x_array[train_index]
        y_train = y_array[train_index]
        x_test = x_array[test_index]
        y_test = y_array[test_index]
#========================================================================================
        r_model = KNeighborsRegressor(n_neighbors=item["n_neighbors"])
#========================================================================================
        r_model.fit(x_train,y_train)
        cnt_predict = r_model.predict(x_test)
        r2.append(r2_score(y_test, cnt_predict))
        mse.append(mean_squared_error(y_test, cnt_predict))
        if best[0] == None or best[0] < r2_score(y_test, cnt_predict):
            best[0] = r2_score(y_test, cnt_predict)
            best[1] = y_test
            best[2] = cnt_predict
    plt.figure(figsize=(9,7))
    plt.plot(best[1],"ro", label="Real")
    plt.plot(best[2],"bo",label="Prediction")
    plt.ylabel("Nº bikes")
    plt.title("Best prediction for SVR Model")
    separator = ", "
    note = "R² Mean: " + str(mean(r2)) + "\n"
    note += "MSE Mean: " + str(mean(mse)) + "\n"

    model_param_text = "n_neighbors: " + str(item["n_neighbors"])

    plt.figtext(x=0.01,y=0.001,s=note)
    plt.figtext(x=0.01,y=0.95,s=model_param_text)
    plt.legend()
    figname = sufixo + "/" + item["ID"] + "_" + sufixo + ".png"
    plt.savefig(figname,format="png",pad_inches=0)
    plt.show()
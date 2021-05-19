import pandas
import numpy as np
import datetime

def GetData(n_lines):
    data_set = pandas.read_csv("day.csv",nrows=n_lines)

    data_set["dteday"] = data_set["dteday"].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").timestamp())
    data_set["season"] = data_set["season"].apply(lambda x: int(x))
    data_set["holiday"] = data_set["holiday"].apply(lambda x: int(x))
    data_set["weekday"] = data_set["weekday"].apply(lambda x: int(x))
    data_set["workingday"] = data_set["workingday"].apply(lambda x: int(x))
    data_set["weathersit"] = data_set["weathersit"].apply(lambda x: int(x))
    data_set["temp"] = data_set["temp"].apply(lambda x: float(x))
    data_set["atemp"] = data_set["atemp"].apply(lambda x: float(x))
    data_set["hum"] = data_set["hum"].apply(lambda x: float(x))
    data_set["windspeed"] = data_set["windspeed"].apply(lambda x: float(x))

    x_array = data_set[["dteday","season","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"]].to_numpy()
    y_array = data_set["cnt"].to_numpy()
    time_array = data_set["dteday"].to_numpy()
    return x_array, y_array, time_array
import pandas
import numpy as np
import datetime

def GetData(cutof_percentage = 0.0, n_lines = 10000):
    data_set = pandas.read_csv("car_prices.csv",sep=";",nrows=n_lines)
    data_set.dropna(inplace=True)
    data_len = int(len(data_set)*(1-cutof_percentage))
    coluns_dict = ["make","model","trim","body","transmission","color","interior"]
    Enum = {}
    for colun in coluns_dict:
        temp_dict = {}
        temp = data_set[colun].unique()
        counter = 1
        for item in temp:
            temp_dict[item] = counter
            counter += 1
        Enum[colun] = temp_dict
        data_set[colun] = data_set[colun].apply(lambda x: Enum[colun][x])

    data_set["condition"] = data_set["condition"].apply(lambda x: float(x.replace(",",".")))         
    #data_set["dteday"] = data_set["dteday"].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").timestamp())

    x_array = data_set[["year","make","model","trim","body","transmission","condition","odometer","color","interior","mmr"]].to_numpy()#["dteday","season","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"]
    x_array = x_array[:data_len]
    y_array = data_set["sellingprice"].to_numpy()
    y_array = y_array[:data_len]
    return x_array, y_array, Enum

def SplitTestAndTrain(new_x_array,y_array,test_percentage = 0.15):
    test_len = int(len(new_x_array)*test_percentage)
    x_train = new_x_array[:-test_len]
    x_test = new_x_array[-test_len:]
    y_train = y_array[:-test_len]
    y_test = y_array[-test_len:]
    return x_train, x_test,y_train,y_test

if __name__ == "__main__":
    GetData()
    
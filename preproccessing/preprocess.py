import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, time
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import r2_score

def fill_nan(data, cat, num):
    pass

def adjust_date(data):
    data['day'] = data['date'].str.split('-|/', expand=True)[0].astype(np.int8) 
    data['month'] = data['date'].str.split('-|/', expand=True)[1].astype(np.int8) 
    data['year'] = data['date'].str.split('-|/', expand=True)[2].astype(np.int16)
    data.drop(columns=['date'], inplace=True)
    return data

def adjust_stop(data):
    data['stop'] = data['stop'].str.split('p', expand=True)[0] + 'p'
    return data

def adjust_route(data):
    data['source'] = data['route'].str.split('\'', expand=True)[3]
    data['destination'] = data['route'].str.split('\'', expand=True)[7]
    data.drop(columns=['route'], inplace=True)
    return data

def adjust_time(data):
    #time_taken_hours = data['time_taken'].str.split('h', expand=True)[0].astype(int)
    #time_taken_minutes = data['time_taken'].str.split('h', expand=True)[1].str.split('m', expand=True)[0] .astype(int)
    #data['time_taken']= time_taken_hours*60 + time_taken_minutes
    #data.drop(columns=['dep_time', 'arr_time', 'year'], inplace=True)
    data['dep_time_hours'] = data['dep_time'].str.split(':', expand=True)[0].astype(int)
    data['dep_time_min'] = data['dep_time'].str.split(':', expand=True)[1].astype(int)
    data['arr_time_hours'] = data['arr_time'].str.split(':', expand=True)[0].astype(int)
    data['arr_tim_min'] = data['arr_time'].str.split(':', expand=True)[1].astype(int)
    # data['time_taken'] = arr_time_tot - dep_time_tot
    data.drop(columns=['dep_time','arr_time', 'year','time_taken'], inplace=True)
    return data
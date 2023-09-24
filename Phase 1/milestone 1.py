import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
import time

warnings.filterwarnings('ignore')

data = pd.read_csv('airline-price-prediction.csv')

y = data['price']
y = y.str.replace(',', '')
y = y.str.replace('.', '')
y = y.astype(float)
data = data.drop(columns=['price'])


data.drop(labels = [127110, 91880], inplace=True)
y.drop(labels = [127110, 91880], inplace=True)

data['day'] = data['date'].str.split('-|/', expand=True)[0].astype(np.int8) 
data['month'] = data['date'].str.split('-|/', expand=True)[1].astype(np.int8) 
data['year'] = data['date'].str.split('-|/', expand=True)[2].astype(np.int16)
data.drop(columns=['date'], inplace=True)

data['stop'] = data['stop'].str.split('p', expand=True)[0] + 'p'

data['source'] = data['route'].str.split('\'', expand=True)[3]
data['destination'] = data['route'].str.split('\'', expand=True)[7]
data.drop(columns=['route'], inplace=True)

data['dep_time_hours'] = data['dep_time'].str.split(':', expand=True)[0].astype(int)
data['dep_time_minutes'] = data['dep_time'].str.split(':', expand=True)[1].astype(int)
data.drop(columns=['dep_time'], inplace=True)

data['arr_time_hours'] = data['arr_time'].str.split(':', expand=True)[0].astype(int)
data['arr_time_minutes'] = data['arr_time'].str.split(':', expand=True)[1].astype(int)
data.drop(columns=['arr_time'], inplace=True)

time_taken_hours = data['time_taken'].str.split('h', expand=True)[0].astype(int)
time_taken_minutes = data['time_taken'].str.split('h', expand=True)[1].str.split('m', expand=True)[0] .astype(int)
data['time_taken']= time_taken_hours*60 + time_taken_minutes

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state = 42)

categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
LE =LabelEncoder()
for col in categorical_columns:
    X_train[col] = LE.fit_transform(X_train[col])
    X_test[col] = LE.transform(X_test[col])
    
SC = StandardScaler()
MMS = MinMaxScaler()
X_train[X_train.columns] = MMS.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = MMS.transform(X_test[X_test.columns])   
y = np.expand_dims(y, axis=1)

X_train.drop(columns=['year','dep_time_hours','dep_time_minutes','arr_time_hours','arr_time_minutes'], inplace=True)
X_test.drop(columns=['year','dep_time_hours','dep_time_minutes','arr_time_hours','arr_time_minutes'], inplace=True)


ticket_data_train = X_train
ticket_data_train['price'] = y_train 

#Feature Selection
#Get the correlation between the features
corr = ticket_data_train.corr()
top_feature = corr.index[abs(corr['price'])>0.2]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = ticket_data_train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)

X_train = X_train[top_feature]
X_test = X_test[top_feature]

while True:
 print("Polynomial Linear Regression press(1)")
 print("Multiable Linear Regression press(2)")
 print("Exit press(0)")
 
 choice=int(input("      Your Choise... "))
 if choice==0:
     break
 if choice==1:
     print("------------------Polynomial Linear Regression------------------")
     poly_features = PolynomialFeatures(degree=4)

     # transforms the existing features to higher degree features.
     X_train_poly = poly_features.fit_transform(X_train)

     # fit the transformed features to Linear Regression
     poly_model = linear_model.LinearRegression()
     startTrain = time.time()
     poly_model.fit(X_train_poly, y_train)
     endTrain = time.time()
     startTest = time.time()

     # predicting on training data-set
     y_train_predicted = poly_model.predict(X_train_poly)
     ypred=poly_model.predict(poly_features.transform(X_test))
     # predicting on test data-set
     prediction = poly_model.predict(poly_features.fit_transform(X_test))
     endTest=time.time()

     print('Co-efficient of Polynomial regression',poly_model.coef_)
     print('Intercept of Polynomail regression model',poly_model.intercept_)
     print("----------------------------------------------------------------")
     print('Mean Square Error ', metrics.mean_squared_error(y_test, prediction))
     print("----------------------------------------------------------------")
     print('Accuracy', r2_score(y_test, prediction))
     print("----------------------------------------------------------------")
     print("Actual Time for Training", endTrain - startTrain)
     print("Actual Time for Prediction", endTest - startTest)
     print("----------------------------------------------------------------")
     actual_price = np.asarray(y_test)[0]
     predicted_value = prediction[0]
     print("Actual_Price = ", str(actual_price))
     print("Predicted_Price = ", str(predicted_value))
     print("----------------------------------------------------------------")


#################################
#################################
 else:
     print("------------------Multiable Linear Regression------------------")
     multi_model = linear_model.LinearRegression()
     startTrain=time.time()
     multi_model.fit(X_train,y_train)
     endTrain=time.time()
     startTest=time.time()

     prediction_multi= multi_model.predict(X_test)
     endTest=time.time()

     print('Co-efficient of Multiable regression',multi_model.coef_)
     print('Intercept of linear regression model',multi_model.intercept_)
    
     print("----------------------------------------------------------------")

     print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction_multi))
     print("----------------------------------------------------------------")

     print("Actual Time for Training",endTrain-startTrain)
     print("Actual Time for Prediction",endTest-startTest)
     print("----------------------------------------------------------------")

     actual_price=np.asarray(y_test)[1]
     predicted_value=prediction_multi[1]
     print("Actual_Price = ",str(actual_price))
     print("Predicted_Price = ",str(predicted_value))
     print("----------------------------------------------------------------")
     print('Accuracy', r2_score(y_test, prediction_multi))
     print("----------------------------------------------------------------")
     print("----------------------------------------------------------------")
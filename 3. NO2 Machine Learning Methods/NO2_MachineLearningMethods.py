#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.model_selection import train_test_split
import sklearn.metrics as mt
from sklearn.svm import SVC
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from math import sqrt
from sklearn.ensemble import IsolationForest

sensor = pd.read_csv("C:/Users/damla/OneDrive/Masaüstü/TUBITAK/5. CALIBRATİIN SENSORS MACHINE LEARNING/YHK1_NO2_tumveriler.csv", sep = ';')

df = pd.DataFrame({'NO2MTHM': sensor["NO2MTHM"], 'WEu_NO2_257': sensor["WEu_NO2_257"], 'Temperature': sensor["Temperature"], 'Humidity': sensor["Humidity"], 'AEu_NO2_257': sensor['AEu_NO2_257'], 'Pressure': sensor['Pressure'], 'WEu_O3_557': sensor['WEu_O3_557'], 'AEu_O3_557': sensor['AEu_O3_557']})

min_max=preprocessing.MinMaxScaler()
col= df.columns
result=min_max.fit_transform(df)
df=pd.DataFrame(result, columns=col) 
df = df.dropna()

X = df[['WEu_NO2_257', 'Temperature', 'Humidity', 'AEu_NO2_257', 'Pressure', 'WEu_O3_557', 'AEu_O3_557']]
Y = df['NO2MTHM']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


iforest = IsolationForest(bootstrap=True,
                          contamination=0.0001, 
                          max_features=7, 
                          max_samples=256, 
                          n_estimators=1000, 
                          n_jobs=-1,
                         random_state=1)
y_pred = iforest.fit_predict(X_train)

X_train_iforest, y_train_iforest = X_train[(y_pred != -1)], Y_train[(y_pred != -1)]

df_train = pd.DataFrame({'NO2MTHM': Y_train, 'WEu_NO2_257': X_train_iforest["WEu_NO2_257"], 'Temperature': X_train_iforest["Temperature"], 'Humidity': X_train_iforest["Humidity"], 'AEu_NO2_257':X_train_iforest["AEu_NO2_257"],'Pressure':X_train_iforest["Pressure"], 'WEu_O3_557': X_train_iforest["WEu_O3_557"], 'AEu_O3_557': X_train_iforest["AEu_O3_557"]})
df_test = pd.DataFrame({'NO2MTHM': Y_test, 'WEu_NO2_257': X_test["WEu_NO2_257"], 'Temperature': X_test["Temperature"], 'Humidity': X_test["Humidity"], 'AEu_NO2_257': X_test["AEu_NO2_257"], 'Pressure': X_test["Pressure"], 'WEu_O3_557': X_test["WEu_O3_557"], 'AEu_O3_557': X_test["AEu_O3_557"]})


# In[9]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

lm = LinearRegression()
lm.fit(X_train_iforest, y_train_iforest)   

y_prediction =  lm.predict(X_test)

score=r2_score(Y_test,y_prediction)
linear_rmse = sqrt(mean_squared_error(Y_test, y_prediction))

print("R2 Score: ", score)
print("RMSE:", linear_rmse)
print("MSE:", mean_squared_error(Y_test, y_prediction))
print('Train skoru: ',lm.score(X_train_iforest,y_train_iforest))
print('Test skoru: ',lm.score(X_test,Y_test))

error = mae(Y_test, y_prediction)
print("Mean absolute error : " + str(error))

df_test["MLR_Pred"] = lm.intercept_ + lm.coef_[0]*df_test["WEu_NO2_257"] + lm.coef_[1]*df_test["Temperature"] + lm.coef_[2]*df_test["Pressure"]+lm.coef_[3]*df_test["Humidity"]+lm.coef_[4]*df_test["AEu_NO2_257"]+lm.coef_[5]*df_test["WEu_O3_557"]+lm.coef_[6]*df_test["AEu_O3_557"]

df_test[["NO2MTHM", "MLR_Pred"]].plot()
plt.xticks(rotation = 20)
sns.lmplot(x = 'NO2MTHM', y = 'MLR_Pred', data = df_test, fit_reg = True, line_kws = {'color': 'orange'}) 


# In[10]:


from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(X_train_iforest,y_train_iforest)

pred = ridgeReg.predict(X_test)

score=r2_score(Y_test,pred)
ridge_rmse = sqrt(mean_squared_error(Y_test, pred))

print("R2 Score: ", score)
print("RMSE:", ridge_rmse)
print("MSE:", mean_squared_error(Y_test, pred))
print('Train skoru: ',ridgeReg.score(X_train_iforest,y_train_iforest))
print('Test skoru: ',ridgeReg.score(X_test,Y_test))

error = mae(Y_test, pred)
print("Mean absolute error : " + str(error))

df_test["Ridge_Pred"] = pred

df_test[["NO2MTHM", "Ridge_Pred"]].plot()
plt.xticks(rotation = 20)
sns.lmplot(x = 'NO2MTHM', y = 'Ridge_Pred', data = df_test, fit_reg = True, line_kws = {'color': 'orange'}) 


# In[11]:


from sklearn.linear_model import ElasticNet

e_net = ElasticNet(alpha = 0.0001)
e_net.fit(X_train_iforest, y_train_iforest)

y_pred_elastic = e_net.predict(X_test)

score=r2_score(Y_test,y_pred_elastic)
elasticNet_rmse = sqrt(mean_squared_error(Y_test, y_pred_elastic))

print("R2 Score: ", score)
print("RMSE:", elasticNet_rmse)
print("MSE:", mean_squared_error(Y_test, y_pred_elastic))

print('Train skoru: ',e_net.score(X_train_iforest,y_train_iforest))
print('Test skoru: ',e_net.score(X_test,Y_test))

error = mae(Y_test, y_pred_elastic)
print("Mean absolute error : " + str(error))

df_test["ElasticNet_Pred"] = y_pred_elastic

df_test[["NO2MTHM", "ElasticNet_Pred"]].plot()
plt.xticks(rotation = 20)
sns.lmplot(x = 'NO2MTHM', y = 'ElasticNet_Pred', data = df_test, fit_reg = True, line_kws = {'color': 'orange'}) 


# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

model = RandomForestRegressor(n_estimators=400, max_depth = 30, random_state=42)

rf = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
rf.fit(X_train, Y_train)

mse = mean_squared_error(rf.predict(X_test),Y_test)
rmse = np.sqrt(mse)
score=r2_score(Y_test,rf.predict(X_test))
print("R2 Score: ", score)
print('Mean Squared Error (MSE): %.2f' % mse)
print('Root Mean Squared Error (RMSE): %.2f' % rmse)
from sklearn.metrics import mean_absolute_error as mae

error = mae(Y_test, rf.predict(X_test))
print("Mean absolute error : " + str(error))

print('Train skoru: ',rf.score(X_train,Y_train))
print('Test skoru: ',rf.score(X_test,Y_test))

df_test["RF_Pred"] = rf.predict(X_test)
# Plot linear
df_test[["NO2MTHM", "RF_Pred"]].plot()
plt.xticks(rotation = 20)

# Plot regression
sns.lmplot(x = 'NO2MTHM', y = 'RF_Pred', data = df_test, fit_reg = True, line_kws = {'color': 'orange'}) 


# In[13]:


from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, Y, test_size=0.33, random_state=42)
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)
poly_reg_y_predicted = poly_reg_model.predict(X_test)

score=r2_score(y_test,poly_reg_y_predicted)
poly_reg_model_rmse = sqrt(mean_squared_error(y_test, poly_reg_y_predicted))

print("R2 Score: ", score)
print("RMSE:", poly_reg_model_rmse)
print("MSE:", mean_squared_error(y_test, poly_reg_y_predicted))
error = mae(y_test, poly_reg_y_predicted)
print("Mean absolute error : " + str(error))
print('Train skoru: ',poly_reg_model.score(X_train,y_train))
print('Test skoru: ',poly_reg_model.score(X_test,y_test))

df_test["MLR_pl_Pred"] = poly_reg_y_predicted
sns.lmplot(x = 'NO2MTHM', y = 'MLR_pl_Pred', data = df_test, fit_reg = True, line_kws = {'color': 'orange'}) 


# In[14]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgbm
import xgboost as xg

def boost_models(x):
    
    regr_trans = TransformedTargetRegressor(regressor=x, transformer=QuantileTransformer(output_distribution='normal'))
    regr_trans.fit(X_train, Y_train)
    yhat = regr_trans.predict(X_test)
    algoname= x.__class__.__name__
    return algoname, round(r2_score(Y_test, yhat),3), round(np.sqrt(mean_squared_error(Y_test, yhat)),2), round(mean_squared_error(Y_test, yhat),2), round(mae(Y_test, pred),2)

algo=[GradientBoostingRegressor(), lgbm.LGBMRegressor()]
score=[]
for a in algo:
    score.append(boost_models(a))

pd.DataFrame(score, columns=['Model', 'Score',  'RMSE', 'MSE', 'MAE'])


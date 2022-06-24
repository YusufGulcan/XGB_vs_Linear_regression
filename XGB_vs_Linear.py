import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

df=pd.read_csv(r'C:\Users\yusuf\Desktop\Workspace url\Trendyol Project/df_trendyol.csv',usecols=['Model','Brand','Price','CPU','RAM','Storage','Operating System','Camera Resolution','Screen Size'])
df.head(3)

df.dropna(how='any', subset=['Model','Brand','CPU','RAM','Storage','Operating System','Camera Resolution','Screen Size'],inplace=True)

df.CPU.replace({'1.5-2.0':1.8,'2.0-2.5': 2.3,'2.5-3.2':2.8, '0.5-1.0':0.8, '1.0-1.5':1.3},inplace=True)

df = pd.get_dummies(df,columns=['Operating System'])

x = df.iloc[:,3:]
y = df.iloc[:,2:3]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 0)

reg = linear_model.LinearRegression()


result = reg.fit(x_train,y_train)

coefficients = reg.coef_
print(f'coefficients: {coefficients}')
interception = reg.intercept_
print(f'interception: {interception}')

y_pred = reg.predict(x_test)

conc=pd.DataFrame({'act':np.squeeze(y_test,1),
             'predicted':np.squeeze(y_pred,1)})


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_pred,y_test))

plt.figure(figsize=(10,6))
plt.scatter(conc.act,conc.predicted,color='g', label = 'Linear Regression Prediction')
plt.plot(conc.act,conc.act,color='r',label='The Real data')
plt.legend()




model1 = XGBRegressor()
model1.fit(x_train,y_train)

y_pred1=model1.predict(x_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_test, y_pred1))
print('R2 score:', metrics.r2_score(y_pred1,y_test))  #R2 score shows that the 97.3 percent of the model can be explained by the coefficients


plt.figure(figsize=(12,8))
plt.plot(y_test,y_test)
plt.scatter(y_test,y_pred1, color = 'g', label = 'XGBooster Regression Model')
plt.scatter(y_test,y_pred, color = 'r', label = 'Linear Regression Model')
plt.xlabel('Price')
plt.legend()
plt.show()


def get_metrics(a,b):        # Function to get all metrics into a list
    MEA = metrics.mean_absolute_error(a,b)
    MSE = metrics.mean_squared_error(a,b)
    RMSE= np.sqrt(metrics.mean_squared_error(a,b))
    r2 = metrics.r2_score(a,b)
    return [MEA,MSE,RMSE,r2]
get_metrics(y_pred,y_test)

metrics = {'XGBoost' : get_metrics(y_test,y_pred1),      # Dictionary
        'Linear Regression' : get_metrics(y_test,y_pred)}

comparison = pd.DataFrame(metrics,index=['MEA','MSE','RMSE','R2'])  # DataFrame
comparison =comparison.transpose()
print(comparison)

#I created a function to gather all metrics into a list which
# I used to create a pandas data frame so we can see the comparison
# between the success rates of the models. To conclude,
# the XGBoost model is the winner by far in every metric.
# Our numeric data and visualization confirm this.
# It was kind of expected but I wanted proof. Thank you for reading all the way here.





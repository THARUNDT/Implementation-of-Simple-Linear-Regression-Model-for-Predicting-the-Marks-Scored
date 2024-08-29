# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THARUN D
RegisterNumber:  212223240167
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/student_scores.csv")
df

x = df.iloc[:,:-1].values
print("X-Values:",x)
y = df.iloc[:,1].values
print("Y-Values:",y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
print("X-Training Data:",x_train)
print("X-Testing Data:",x_test)
print("Y-Training Data:",y_train)
print("Y-Testing Data:",y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Y-Predited:",y_pred)
print("Y-Testing",y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE  = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE  = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![ml 1](https://github.com/user-attachments/assets/7c48ab81-1d1a-4163-8fed-01931f5c0085)

![ml2](https://github.com/user-attachments/assets/ba45cd9d-41e3-4d1f-b5f9-ad534e279ec9)

![ml3](https://github.com/user-attachments/assets/5feae8c9-880e-47e8-bc55-70e23b59cc1e)

![ml4](https://github.com/user-attachments/assets/8b3a5b07-dfef-4d61-9332-8c91e1378ec7)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

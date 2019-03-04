# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#LINEAR REGRESSION

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error
print("mean_absolute_error : ",mean_absolute_error(y_test, y_pred))


from sklearn.metrics import mean_squared_error
print("mean_squared_error : ",mean_squared_error(y_test, y_pred))
print("mean_squared_error : ",np.sqrt(mean_squared_error(y_test, y_pred)))

#RANDOM FOREST REGRESSION


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Random forest to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor1.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor1.predict(X_test)

from sklearn.metrics import mean_absolute_error
print("mean_absolute_error : ",mean_absolute_error(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print("mean_squared_error : ",mean_squared_error(y_test, y_pred))
print("mean_squared_error : ",np.sqrt(mean_squared_error(y_test, y_pred)))

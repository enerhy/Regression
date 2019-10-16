# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data # Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#adding a b0 to the regression
X_train_with_bo = np.append(arr = np.ones((40, 1)).astype(int), values = X_train , axis = 1)
X_test_with_bo =  np.append(arr = np.ones((10, 1)).astype(int), values = X_test , axis = 1)

"""Backwards Elimination with with p-Value and Adjusted R-Squared"""
def backwardselimination_R(X, SL):
    numVars = len(X[0])
    temp = np.zeros((40,6)).astype(int)
    for i in range(0, numVars):             #For each value i
        regressor_OLS = sm.OLS(y_train, X).fit()    # we fit a regressor to y_train with the last X 
                                                    #(changed from previous itterations)
        maxVar = max(regressor_OLS.pvalues).astype(float)   #we extract the maximum p-value of remaining variables
        adjR_before = regressor_OLS.rsquared_adj.astype(float) #extract the adjR-squared from this regression
        
        if maxVar > SL:                      #if the hightest p-value > 0.05, than:
            for j in range(0, numVars - i):  #looping to find the variable with the max pvalue
                if (regressor_OLS.pvalues[j].astype(float) == maxVar): # once the variable with the max pvalue is found than:
                    temp[:,j] = X[:,j]
                    X = np.delete(X, j, 1)
                    tmp_regressor = sm.OLS(y_train, X).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((X, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print(regressor_OLS.summary())
    return X

SL = 0.05
X_opt = X_train_with_bo[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardselimination_R(X_opt, SL)
        



"""Backwards Elimination with P-value"""

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X_train_with_bo[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_2 = LinearRegression()
regressor_2.fit(X_Modeled, y_train)

# Predicting the Test set results
X_opt_test = X_test_with_bo[:, [0, 3]]
y_pred_opt = regressor_2.predict(X_opt_test)

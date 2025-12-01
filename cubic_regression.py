import numpy as np   # import numpy libaray
from sklearn.preprocessing import PolynomialFeatures # import PolynomialFeatures
from sklearn.linear_model import LinearRegression # import model
from sklearn.metrics import mean_squared_error,r2_score # import metrics
from sklearn.model_selection import train_test_split # import train_test_split to split the data for training nad testing
import joblib

set1=list(i for i in range(1,11)) # generate sample dataset of numbers from 1 to 10
set2=[x**3 for x in set1]  # generate sample dataset of numbers raised to the power of 3

X=np.array(set1).reshape(-1,1) # convert them to numpy array for faster calculations
Y=np.array(set2).reshape(-1,1)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0) # split the data for training and testing
poly=PolynomialFeatures(degree=3) # create polynomial features
X_train=poly.fit_transform(X_train) # fit the polynomial features
model=LinearRegression() # create model
model.fit(X_train,Y_train) # fit the model
X_test=poly.transform(X_test)
Y_pred=model.predict(X_test) # predict the values
ms=mean_squared_error(Y_test,Y_pred) # calculate mean squared error
r2=r2_score(Y_test,Y_pred) # calculate r2 score
print("Mean Squared Error: " ,ms)
print("R2 Score: ",r2)
joblib.dump(model,"cubic.pkl")  # save the model
joblib.dump(poly,"CubicFeatures.pkl") # save the polynomial features


inx=float(input("Enter any number: "))
inx=np.array(inx).reshape(-1,1)
inx=poly.transform(inx)

iny=model.predict(inx)
print("Predicted value: ",round(iny[0][0],2))



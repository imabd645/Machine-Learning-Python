import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Create data: set1 = [1..10], set2 = squares of these numbers
set1 = list(i for i in range(1, 11))
set2 = [x * x for x in set1]

# Convert lists to numpy arrays and reshape for sklearn
X = np.array(set1).reshape(-1, 1)
Y = np.array(set2).reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create polynomial features (degree 2 for quadratic curve)
poly = PolynomialFeatures(degree=2)

# Fit polynomial transformation on training data
X_train = poly.fit_transform(X_train)

# Create and train the linear regression model using polynomial features
model = LinearRegression()
model.fit(X_train, Y_train)

# Transform test data using the same polynomial transformation
X_test = poly.transform(X_test)

# Predict using the trained model
Y_pred = model.predict(X_test)

# Calculate evaluation metrics
ms = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Print model accuracy
print("Mean Squared Error: ", ms)
print("R2 Score: ", r2)

# Save both the model and polynomial transformer for later use
joblib.dump(model, "Polynomial.pkl")
joblib.dump(poly, "polyFeatures.pkl")

# Take a user input number for prediction
inx = float(input("Enter any number: "))

# Convert input to numpy array and apply polynomial transformation
inx = np.array(inx).reshape(-1, 1)
inx = poly.transform(inx)

# Predict final output
iny = model.predict(inx)

# Display predicted value
print("Predicted value: ", iny[0][0])

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Create dataset: numbers 1–14
set1 = list(i for i in range(1, 15))

# Compute cube of each number
set2 = [x**3 for x in set1]

# Convert to numpy arrays (X as 2D, Y as 1D)
X = np.array(set1).reshape(-1, 1)
Y = np.array(set2).ravel()

# Split data into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create and train the Gradient Boosting Regressor model
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)

# Predict values for test set
Y_pred = model.predict(X_test)

# Calculate error metrics
ms = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Print model performance
print("Mean Squared Error: ", ms)
print("R2 Score: ", r2)

# Save the trained model to a file
joblib.dump(model, "Gradient_Boosting.pkl")

# Take user input to predict a new value
inx = float(input("Enter any number: "))

# Reshape input to match model requirements
inx = np.array(inx).reshape(-1, 1)

# Predict the output for the input number
iny = model.predict(inx)

# Print the prediction (rounded to 2 decimals)
print("Predicted Value: ", round(iny[0], 2))

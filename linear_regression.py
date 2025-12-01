import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Create dataset: X = 1..10
set1 = list(i for i in range(1, 11))

# Y = 2 * X (simple linear relationship)
set2 = [2 * x for x in set1]

# Convert lists to numpy arrays and reshape for sklearn
X = np.array(set1).reshape(-1, 1)
Y = np.array(set2).reshape(-1, 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values for the test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save trained model to file
joblib.dump(model, "model.pkl")

# Take user input
inx = int(input("Enter the value of x:"))

# Predict output for the given input
iny = model.predict([[inx]])

# Display the predicted value
print("Predicted value:", iny[0][0])

import joblib
import numpy

# Load the trained polynomial regression model
model = joblib.load("cubic.pkl")

# Load the polynomial feature transformer used during training
poly = joblib.load("CubicFeatures.pkl")

# Take numeric input from the user
x = float(input("Enter any number: "))

# Convert the input into a numpy array and reshape for model compatibility
x = numpy.array(x).reshape(-1, 1)

# Apply the same polynomial transformation used during training
x = poly.transform(x)

# Predict the output using the loaded model
y = model.predict(x)

# Print the predicted value (rounded to 2 decimal places)
print("Predicted Value: ", round(y[0][0], 2))

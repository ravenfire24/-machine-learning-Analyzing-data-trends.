import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample sales data 
data = {
    "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Marketing Spend": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500],
    "Holiday Season": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    # Adding noise to the sales data to make predictions harder
    "Sales": [2000, 2900, 2450, 2650, 3900, 3450, 3550, 4400, 3950, 4100, 4950, 4600]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Features (independent variables) and target (dependent variable)
X = df[["Month", "Marketing Spend", "Holiday Season"]]
y = df["Sales"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Calculating accuracy (R^2 score)
accuracy = r2_score(y_test, y_pred) * 100
print(f"Model achieved an accuracy of {accuracy:.2f}% on test data.")

# sample prediction
sample_input = pd.DataFrame({"Month": [13], "Marketing Spend": [7000], "Holiday Season": [0]})
predicted_sales = model.predict(sample_input)
print(f"Predicted Sales for Month 13: ${predicted_sales[0]:.2f}")

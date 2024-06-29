import pandas as pd


# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv(r'C:\Users\jhurl\Downloads\cpy\nba\nba_data.csv')

df = df.iloc[:, :-1]  # Remove the last column from the dataframe

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Assuming 'height', 'weight', and 'position' columns exist in df

# Prepare data for 'height vs position'
X_height = df[['HEIGHT']]
y_position = df['POSITION']

# Prepare data for 'weight vs position'
X_weight = df[['WEIGHT']]

# Linear regression for 'height vs position'
model_height = LinearRegression()
X_train_height, X_test_height, y_train_height, y_test_height = train_test_split(X_height, y_position, test_size=0.2, random_state=42)
model_height.fit(X_train_height, y_train_height)
r2_height = model_height.score(X_test_height, y_test_height)

# Linear regression for 'weight vs position'
model_weight = LinearRegression()
X_train_weight, X_test_weight, y_train_weight, y_test_weight = train_test_split(X_weight, y_position, test_size=0.2, random_state=42)
model_weight.fit(X_train_weight, y_train_weight)
r2_weight = model_weight.score(X_test_weight, y_test_weight)

print(f"R^2 for height vs position: {r2_height}")
print(f"R^2 for weight vs position: {r2_weight}")

import numpy as np

# Assuming the following variables are defined based on the context of nba_2.py:
# model_height, model_weight, X_test_height, X_test_weight, y_test_height, y_test_weight

# Step 1: Random Sampling
num_samples = 100  # Adjust as needed
indices_height = np.random.choice(X_test_height.index, size=num_samples, replace=False)
indices_weight = np.random.choice(X_test_weight.index, size=num_samples, replace=False)

# Step 2: Predict Positions
predictions_height = model_height.predict(X_test_height.loc[indices_height])
predictions_weight = model_weight.predict(X_test_weight.loc[indices_weight])

# Step 3: Round Predictions
rounded_predictions_height = np.round(predictions_height * 2) / 2
rounded_predictions_weight = np.round(predictions_weight * 2) / 2

# Step 4 & 5: Compare Predictions to Actual Values and Count Correct Predictions
actual_positions_height = y_test_height.loc[indices_height]
actual_positions_weight = y_test_weight.loc[indices_weight]

correct_predictions_height = np.sum(rounded_predictions_height == actual_positions_height)
correct_predictions_weight = np.sum(rounded_predictions_weight == actual_positions_weight)

# Output the number of correct predictions
print(f"Correct predictions using height model: {correct_predictions_height} out of {num_samples}")
print(f"Correct predictions using weight model: {correct_predictions_weight} out of {num_samples}")
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

import numpy as np

# Step 1: Randomly Predict Positions
# Assuming positions are integers in a known range, for example, 1 to 5
random_predictions = np.random.randint(1, 6, size=num_samples)  # Adjust 1, 6 to the actual range of positions

# Step 2: Count Correct Random Predictions
correct_random_predictions = np.sum(random_predictions == actual_positions_height.values)

# Step 3: Predict 0 for Player's Positions
zero_predictions = np.zeros(num_samples, dtype=int)

# Step 4: Count Correct Zero Predictions
correct_zero_predictions = np.sum(zero_predictions == actual_positions_height.values)

# Output the number of correct predictions
print(f"Correct random predictions: {correct_random_predictions} out of {num_samples}")
print(f"Correct zero predictions: {correct_zero_predictions} out of {num_samples}")

# Calculate the mean of the actual positions
mean_actual_positions = np.mean(actual_positions_height.values)

# Calculate SS_tot
SS_tot = np.sum((actual_positions_height.values - mean_actual_positions) ** 2)

# Since the model predicts 0, SS_res is just the sum of squared actual values
SS_res = np.sum(actual_positions_height.values ** 2)

# Calculate R^2
R_squared = 1 - (SS_res / SS_tot)

print(f"R^2 value for the model constantly predicting 0: {R_squared}")
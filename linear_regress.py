# 1. Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. Create Dummy Data
# Let's assume we have data on:
# - Opposition Team (as integer label)
# - Match Location (home=1, away=0)
# - Recent Form: Average of last 5 innings
# - Previous Match Score
# - Target variable: Actual Score in this match

np.random.seed(42)  # for reproducibility

# Create 100 samples
data = {
    'Opposition_Team': np.random.randint(1, 6, 100),  # say 5 teams
    'Match_Location': np.random.randint(0, 2, 100),   # home or away
    'Recent_Form': np.random.uniform(20, 80, 100),    # avg of last 5 innings
    'Previous_Score': np.random.uniform(0, 120, 100), # last match score
    'Score': np.random.uniform(0, 150, 100)           # actual score (target)
}

df = pd.DataFrame(data)

# 3. Prepare features (X) and target (y)
X = df[['Opposition_Team', 'Match_Location', 'Recent_Form', 'Previous_Score']]
y = df['Score']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict on test set
y_pred = model.predict(X_test)

# 7. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
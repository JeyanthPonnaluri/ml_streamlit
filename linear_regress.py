# linear_regress.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create Dummy Data
np.random.seed(42)

data = {
    'Opposition_Team': np.random.randint(1, 6, 100),
    'Match_Location': np.random.randint(0, 2, 100),
    'Recent_Form': np.random.uniform(20, 80, 100),
    'Previous_Score': np.random.uniform(0, 120, 100),
    'Score': np.random.uniform(0, 150, 100)
}

df = pd.DataFrame(data)

# Prepare features (X) and target (y)
X = df[['Opposition_Team', 'Match_Location', 'Recent_Form', 'Previous_Score']]
y = df['Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Optional: print when running standalone
if __name__ == "__main__":
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Example
    new_data = pd.DataFrame({
        'Opposition_Team': [2],
        'Match_Location': [1],
        'Recent_Form': [65],
        'Previous_Score': [75]
    })

    predicted_score = model.predict(new_data)
    print(f"Predicted Score in next innings: {predicted_score[0]:.2f}")

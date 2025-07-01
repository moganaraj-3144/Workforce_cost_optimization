from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
from src.preprocess.preprocessing import load_and_prepare_data, get_preprocessor
from config import DATA_PATH, ML_MODEL_PATH


df = load_and_prepare_data(DATA_PATH)

df = df.drop(columns=['lat', 'lng', 'remote_vs_office_cost_ratio',
                      'In-Office Expenses', 'Remote Work Expenses',
                      ])
X = df.drop(columns=['Total Remote salary'])
y = df['Total Remote salary']

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
}

best_model = None
best_score = -np.inf
best_name = ""

preprocessor = get_preprocessor()

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    avg_score = np.mean(scores)
    print(f"{name} R² Score: {avg_score:.4f}")

    if avg_score > best_score:
        best_score = avg_score
        best_model = pipeline
        best_name = name

print(f"\nBest Model: {best_name} with R² = {best_score:.4f}")

# Fit best model on entire data
best_model.fit(X, y)
joblib.dump(best_model, ML_MODEL_PATH)

import pandas as pd
import joblib
from spreg import OLS
from libpysal.weights import KNN
from src.preprocess.preprocessing import load_and_prepare_data
from config import DATA_PATH, SPATIAL_MODEL_PATH

# DATA_PATH = "data/hiring_data.csv"
# SPATIAL_MODEL_PATH = "artifacts/spatial_model.pkl"

df = load_and_prepare_data(DATA_PATH)
y = df['Total Remote salary'].values.reshape(-1, 1)
X = df[['Salary_in_office', 'Cost of Living Index', 'In-Office Expenses']].values

coords = list(zip(df['lat'], df['lng']))
w = KNN.from_array(coords, k=3)
model = OLS(y, X, w=w, name_y='Total Remote salary', name_x=['Salary_in_office', 'COL', 'In-Office'])

joblib.dump(model, SPATIAL_MODEL_PATH)

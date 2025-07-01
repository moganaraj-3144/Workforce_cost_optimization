from scipy.optimize import linprog
import joblib
import pandas as pd
from src.preprocess.preprocessing import load_and_prepare_data
from config import DATA_PATH, OPT_MODEL_PATH

# DATA_PATH = "data/hiring_data.csv"
# OPT_MODEL_PATH = "artifacts/opt_model.pkl"

df = load_and_prepare_data(DATA_PATH)

# Example cost objective: minimize remote or in-office salary under budget
costs = df[['Total In-office salary', 'Total Remote salary']].min(axis=1)
constraints = [sum(costs) <= 1_000_000]

opt_model = {"costs": costs, "df": df}
joblib.dump(opt_model, OPT_MODEL_PATH)

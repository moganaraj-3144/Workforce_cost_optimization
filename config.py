# config.py
# import os

# DATA_PATH = "/Users/mogana/myProjects/Workforce_cost_optimization/Notebook code files/Data/Final_Dataset.csv"
# ML_MODEL_PATH = os.path.join('artifacts', 'ml_model.pkl')
# SPATIAL_MODEL_PATH = os.path.join('artifacts', 'spatial_model.pkl')
# OPT_MODEL_PATH = os.path.join('artifacts', 'opt_model.pkl')

# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "Final_Dataset.csv")
ML_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "ml_model.pkl")
SPATIAL_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "spatial_model.pkl")
OPT_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "opt_model.pkl")

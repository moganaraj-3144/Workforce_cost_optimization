import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import DATA_PATH, ML_MODEL_PATH


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

def get_preprocessor():
    categorical = ['Role', 'Sector', 'Experience', 'City', 'State']
    numerical = ['Salary_in_office', 'Salary_remote', 'Office Rent', 'Annual Commute Cost',
                 'Avg internet monthly payment', 'Electricity Price (cents per kWh)', 
                 'Cost of Living Index', 'National_Avg_Salary']
    # numerical = ['Office Rent', 'Annual Commute Cost',
    #              'Avg internet monthly payment', 'Electricity Price (cents per kWh)', 
    #              'Cost of Living Index', 'National_Avg_Salary']
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', StandardScaler(), numerical)
    ])
    return preprocessor

# Workforce Cost Optimization

A data-driven AI solution to optimize hiring decisions by comparing costs between remote and in-office workers across U.S. cities. This project uses machine learning, spatial regression, geospatial clustering, and linear optimization to recommend the most cost-effective hiring strategy based on job role, location, experience, and budget.


## Project Objectives

- Analyze hiring costs for remote vs. in-office employees.
- Use ML to predict total workforce costs across cities.
- Perform geospatial and spatial regression analysis to capture regional salary variation.
- Recommend optimal hiring strategy using linear programming.


## Datasets Used

| Dataset | Description |
|--------|-------------|
| **Job Listings (Scraped)** | Salaries, roles, locations, benefits for remote/in-office jobs. |
| **O*NET & OEWS** | Skill, role, and wage data across the U.S. |
| **Office Market Data** | Office rent, space allocation, regional trends. |
| **Remote Work Expenses** | Internet, electricity, rent, mobile usage for remote settings. |
| **Commuting Costs** | Regional commute costs from the U.S. Census Bureau. |


## Methodologies

1. **Geospatial Clustering**  
   Groups cities by salary/rent patterns to uncover hiring cost clusters.

2. **Spatial Regression (GWR)**  
   Models local salary variations across geography as a fallback model.

3. **Machine Learning Models**  
   Predicts total cost using Random Forest, XGBoost, etc.  
   ➤ Best model: Random Forest (R² ≈ 0.82)

4. **Linear Programming**  
   Optimizes remote vs. in-office hiring based on user’s budget.


## Key Features

- Input: Role, Experience, Budget, and Location
- Output: Predicted cost and recommended hiring strategy
- AI Models: ML for cost prediction, Spatial Regression for fallback
- Streamlit Dashboard: Interactive UI to experiment with different inputs
- Tableau Dashboard: Visualizes geospatial clustering of hiring costs across U.S. cities using interactive map.


## Tech Stack

- **Language**: Python
- **Libraries**: Scikit-learn, XGBoost, GWR, PySAL, Streamlit, Pandas, Geopandas
- **IDE**: Visual Studio Code
- **Deployment**: Streamlit
- **Version Control**: Git + GitHub


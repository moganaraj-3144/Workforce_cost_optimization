def predict_salary(model, df, role, exp, location):
    # Simplified matching
    match = df[(df['Role'] == role) & (df['Experience'] == exp)]
    if not match.empty:
        return model.predict(match.drop(['Total Remote salary'], axis=1))[0]
    else:
        return model.predict(df.drop(['Total Remote salary'], axis=1).mean().values.reshape(1, -1))[0]

def recommend_strategy(opt_model, budget):
    df = opt_model["df"]
    costs = opt_model["costs"]
    min_cost = costs.min()
    return "Remote" if df['Total Remote salary'].mean() < df['Total In-office salary'].mean() else "In-Office"

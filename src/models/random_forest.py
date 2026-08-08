from sklearn.ensemble import RandomForestRegressor

def train_random_forest(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=None,
    random_state=42,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train,y_train)

    return model


def predict_random_forest(model,X):
    return model.predict(X)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def train_ridge(
    X_train,
    y_train,
    alpha=1.0
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train
    )
    model = Ridge(
        alpha=alpha
    )
    model.fit(
        X_train_scaled,
        y_train
    )
    return model, scaler

def predict_ridge(
    model,
    scaler,
    X
):
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)
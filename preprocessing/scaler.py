def StandartScaler(X):
    return (X - X.mean()) / X.std()
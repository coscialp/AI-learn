def StandartScaler(X, mean=None, std=None):
    if mean == None and std == None:
        return (X - X.mean()) / X.std()
    elif mean == None:
        return (X - X.mean()) / std
    elif std == None:
        return (X - mean) / X.std()
    else:
        return (X - mean) / std
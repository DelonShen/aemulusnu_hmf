import numpy as np

def scaleToRedshift(a):
    return 1/a-1

def redshiftToScale(z):
    return 1/(1+z)


class Normalizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)

    def transform(self, X):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")

        return (X - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, X_normalized):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")
        return X_normalized * (self.max_val - self.min_val) + self.min_val


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted. Call fit() first.")
        return (X - self.mean) / self.std

    def inverse_transform(self, X_std):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fitted. Call fit() first.")
        return X_std * self.std + self.mean


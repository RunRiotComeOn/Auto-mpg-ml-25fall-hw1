import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Define class of my polynomial regression solver

class MultiPolyRegression:
    def __init__(self, degree=1):
        self.degree = degree
        self.coefficients = None

    def _create_poly_features(self, X): # Mainly modified this
        X = np.array(X)
        n_samples, n_features = X.shape

        if self.degree == 0:
            return np.ones((n_samples, 1))
        
        elif self.degree == 1:
            return np.column_stack([np.ones(n_samples), X])
        
        elif self.degree == 2:
            terms = [np.ones(n_samples)]
            
            for i in range(n_features):
                terms.append(X[:, i])
            
            for i in range(n_features):
                terms.append(X[:, i] ** 2)
            
            return np.column_stack(terms)
    
    def fit(self, x, y):
        y = np.array(y) 
        X = self._create_poly_features(x)

        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

        return self
    
    def predict(self, x):
        X = self._create_poly_features(x)
        y_pred = X @ self.coefficients

        return y_pred
    
    def mse(self, x, y):
        y = np.array(y)
        y_pred = self.predict(x)
        
        mse = np.mean((y - y_pred)**2)
        return mse
    
# Solve question 4

def main():
    df = pd.read_csv("./auto-mpg/auto-mpg-clean.csv")
    df.drop(columns='car name', inplace=True)

    # Split train and test sets
    n_train = 292

    X_train = df.iloc[:n_train].drop(columns = 'mpg')
    y_train = df.iloc[:n_train]['mpg'].values

    X_test = df.iloc[n_train:].drop(columns = 'mpg')
    y_test = df.iloc[n_train:]['mpg'].values

    # Fit model and predict
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'model year', 'origin']

    x_train = X_train[features].values
    x_test = X_test[features].values

    results = {}

    models = {}
    
    for degree in range(3):
        model = MultiPolyRegression(degree=degree)
        model.fit(x_train, y_train)
        train_mse = model.mse(x_train, y_train)
        test_mse = model.mse(x_test, y_test)
        n_terms = len(model.coefficients)

        results[degree] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'n_terms': n_terms
        }
        
        models[degree] = model
    
    # Report
    print("\n(a) Training Mean Squared Errors:")
    for degree in [0, 1, 2]:
        print(f"  {degree}th order polynomial: {results[degree]['train_mse']:.4f}")
    
    print("\n(b) Testing Mean Squared Errors:")
    for degree in [0, 1, 2]:
        print(f"  {degree}th order polynomial: {results[degree]['test_mse']:.4f}")

    best_degree = min([0, 1, 2], key=lambda d: results[d]['test_mse'])
    
    print("\nBest Performance:")
    print(f"Best polynomial degree: {best_degree}")
    print(f"Test MSE: {results[best_degree]['test_mse']:.4f}")

    with open('./ckpts/poly_regression_model_degree2.pkl', 'wb') as f:
        pickle.dump(models[2], f)
    
    with open('./ckpts/feature_names.pkl', 'wb') as f:
        pickle.dump(features, f)

if __name__ == "__main__":
    main()
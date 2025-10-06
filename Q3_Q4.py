import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define class of my polynomial regression solver

class SinglePolyRegression:
    def __init__(self, degree=1):
        self.degree = degree
        self.coefficients = None

    def _create_poly_features(self, x):
        n = len(x)
        X = np.zeros((n, self.degree + 1))

        for i in range(self.degree + 1):
            X[:, i] = x**i

        return X
    
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

    X_train_all_feature = df.iloc[:n_train].drop(columns = 'mpg')
    y_train = df.iloc[:n_train]['mpg'].values

    X_test_all_feature = df.iloc[n_train:].drop(columns = 'mpg')
    y_test = df.iloc[n_train:]['mpg'].values

    # Fit model and predict
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'model year', 'origin']
    
    all_results = []

    for feature in features:
        x_train = X_train_all_feature[feature].values
        x_test = X_test_all_feature[feature].values

        feature_results = {
            'feature': feature,
            'models': {},
            'train_mse': {},
            'test_mse': {}
        }

        for degree in range(4):
            model = SinglePolyRegression(degree=degree)

            model.fit(x_train, y_train)

            train_mse = model.mse(x_train, y_train)
            test_mse = model.mse(x_test, y_test)

            feature_results['models'][degree] = model
            feature_results['train_mse'][degree] = train_mse
            feature_results['test_mse'][degree] = test_mse
        
        all_results.append(feature_results)

    # Plot
    colors = ['blue', 'green', 'orange', 'red']
    
    for feature_results in all_results:
        feature = feature_results['feature']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x_test = X_test_all_feature[feature].values
        
        ax.scatter(x_test, y_test, alpha=0.6, s=40, color='black', 
                  label='Test Data', zorder=5, edgecolors='white', linewidth=0.5)
        
        x_line = np.linspace(x_test.min(), x_test.max(), 300)
        
        for degree in range(4):
            model = feature_results['models'][degree]
            y_line = model.predict(x_line)
            test_mse = feature_results['test_mse'][degree]
            
            ax.plot(x_line, y_line, color=colors[degree], linewidth=2.5,
                   label=f'Degree {degree} (MSE={test_mse:.2f})', alpha=0.8)
        
        ax.set_xlabel(feature, fontsize=13, fontweight='bold')
        ax.set_ylabel('MPG', fontsize=13, fontweight='bold')
        ax.set_title(f'Polynomial Regression: {feature} → MPG\n(Testing Set)', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        filename = f'./plots/q4_{feature.replace(" ", "_")}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

    # Analyze results
    print("Training MSE:")
    for feature_results in all_results:
        feature = feature_results['feature']
        print(f"{feature:<15}", end="")
        for degree in range(4):
            mse = feature_results['train_mse'][degree]
            print(f"{mse:<15.4f}", end="")
            print()

    print("\nTesting MSE:") 
    for feature_results in all_results:
        feature = feature_results['feature']
        print(f"{feature:<15}", end="")
        for degree in range(4):
            mse = feature_results['test_mse'][degree]
            print(f"{mse:<15.4f}", end="")
            print()
    
    best_mse = float('inf')
    best_feature = None
    best_degree = None
    
    for feature_results in all_results:
        for degree in range(4):
            test_mse = feature_results['test_mse'][degree]
            if test_mse < best_mse:
                best_mse = test_mse
                best_feature = feature_results['feature']
                best_degree = degree
    
    print(f"\nBest degree: {best_degree}")
    print(f"Most informative feature: {best_feature}")
    print(f"Lowest MSE: {best_mse:.4f}")

if __name__ == "__main__":
    main()
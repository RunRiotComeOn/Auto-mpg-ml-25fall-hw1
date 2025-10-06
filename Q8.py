import pandas as pd
import numpy as np
import pickle
from Q5 import MultiPolyRegression
from Q6_Q7 import categorize_mpg

def get_category_name(category_num):
    categories = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}
    return categories.get(category_num, 'Unknown')

def main():
    specs = {
        'cylinders': 4,
        'displacement': 400,
        'horsepower': 150,
        'weight': 3500,
        'acceleration': 8,
        'model year': 81,
        'origin': 1
    }
    
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'model year', 'origin']
    
    X_new = np.array([[specs[f] for f in features]])
    
    # Load checkpoints to predict

    with open('./ckpts/poly_regression_model_degree2.pkl', 'rb') as f:
        poly_model = pickle.load(f)
    
    predicted_mpg = poly_model.predict(X_new)[0]
    predicted_category_num = categorize_mpg(predicted_mpg)
    predicted_category_name = get_category_name(predicted_category_num)
    
    with open('./ckpts/logistic_regression_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    
    predicted_category_logistic = logistic_model.predict(X_new)[0]
    predicted_category_logistic_name = get_category_name(predicted_category_logistic)
    predicted_proba = logistic_model.predict_proba(X_new)[0]
    
    # Print results
    print(f"Predicted MPG: {predicted_mpg:.2f}")
    print(f"MPG Category (from polynomial): {predicted_category_name}")
    print(f"MPG Category (from logistic regression): {predicted_category_logistic_name}")
    
    print(f"\nPrediction Probabilities:")
    category_names = ['Low', 'Medium', 'High', 'Very High']
    for cat_name, prob in zip(category_names, predicted_proba):
        print(f"  {cat_name:12s}: {prob:.4f}")

if __name__ == "__main__":
    main()
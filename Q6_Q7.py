import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, classification_report, confusion_matrix

def categorize_mpg(mpg):

    if mpg <= 17.00:
        return 0  # Low
    elif mpg <= 22.75:
        return 1  # Medium
    elif mpg <= 29.00:
        return 2  # High
    else:
        return 3  # Very High

def question_6():
    df = pd.read_csv("./auto-mpg/auto-mpg-clean.csv")
    
    df['mpg_category'] = df['mpg'].apply(categorize_mpg)

    # shuffle the data to obtain more balanced categories
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Nevertheless, it didn't change the fact that precise without normalization is better than precise with normalization.
    
    n_train = 292
    
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'model year', 'origin']
    
    X_train = df.iloc[:n_train][features].values
    y_train = df.iloc[:n_train]['mpg_category'].values
    
    X_test = df.iloc[n_train:][features].values
    y_test = df.iloc[n_train:]['mpg_category'].values
    
    # Train Logistic Regression and predict
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_precision = precision_score(y_train, y_train_pred, average='macro') 
    test_precision = precision_score(y_test, y_test_pred, average='macro')

    print("For question 6")    
    print(f"\n(a) Training Set Precision: {train_precision:.4f}")
    print(f"\n(b) Testing Set Precision: {test_precision:.4f}")

    print("Detailed Classification Report (Testing Set, Normalized):")
    target_names = ['Low', 'Medium', 'High', 'Very High']
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    print("Confusion Matrix (Testing Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    with open('logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return X_train, X_test, y_train, y_test


def question_7(X_train_q6, X_test_q6, y_train, y_test):
    # Apply minmaxscaler
    scaler = MinMaxScaler()
    
    X_train_normalized = scaler.fit_transform(X_train_q6)
    X_test_normalized = scaler.transform(X_test_q6)
    
    model_normalized = LogisticRegression(max_iter=1000, random_state=42)
    model_normalized.fit(X_train_normalized, y_train)
    
    y_train_pred_norm = model_normalized.predict(X_train_normalized)
    y_test_pred_norm = model_normalized.predict(X_test_normalized)
    
    train_precision_norm = precision_score(y_train, y_train_pred_norm, average='macro')
    test_precision_norm = precision_score(y_test, y_test_pred_norm, average='macro')
    
    print("For question 6")
    print(f"\n(a) Training Set Precision (Normalized): {train_precision_norm:.4f}")
    print(f"\n(b) Testing Set Precision (Normalized): {test_precision_norm:.4f}")   

    print("Detailed Classification Report (Testing Set, Normalized):")
    target_names = ['Low', 'Medium', 'High', 'Very High']
    print(classification_report(y_test, y_test_pred_norm, target_names=target_names))

    with open('./ckpts/logistic_regression_model_normalized.pkl', 'wb') as f:
        pickle.dump(model_normalized, f)
    
    with open('./ckpts/minmax_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


def main():

    X_train, X_test, y_train, y_test = question_6()
    question_7(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
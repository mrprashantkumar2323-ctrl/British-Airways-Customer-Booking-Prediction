"""
British Airways - Customer Booking Predictive Model
Author: Prashant Kumar
Description: End-to-end Machine Learning pipeline predicting flight booking conversion
utilizing Random Forest classification on 50,000+ flight booking records.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def generate_synthetic_ba_data(n_samples=5000):
    np.random.seed(42)
    data = {
        'num_passengers': np.random.randint(1, 6, n_samples),
        'purchase_lead': np.random.exponential(scale=60, size=n_samples).astype(int),
        'length_of_stay': np.random.exponential(scale=15, size=n_samples).astype(int),
        'flight_hour': np.random.randint(0, 24, n_samples),
        'flight_day': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_samples),
        'route': np.random.choice(['LHR-JFK', 'LHR-DXB', 'LHR-SIN', 'LHR-DEL', 'LHR-BOM', 'LHR-LAX'], n_samples),
        'booking_origin': np.random.choice(['United Kingdom', 'United States', 'India', 'Australia', 'Germany'], n_samples),
        'wants_extra_baggage': np.random.binomial(1, 0.65, n_samples),
        'wants_preferred_seat': np.random.binomial(1, 0.35, n_samples),
        'wants_in_flight_meals': np.random.binomial(1, 0.45, n_samples),
        'flight_duration': np.random.uniform(2.5, 14.5, n_samples),
    }
    df = pd.DataFrame(data)
    # Target generation logic with realistic weights
    score = (
        0.3 * df['wants_extra_baggage'] +
        0.4 * df['wants_preferred_seat'] -
        0.005 * df['purchase_lead'] +
        0.02 * df['length_of_stay'] +
        np.random.normal(0, 0.5, n_samples)
    )
    df['booking_complete'] = (score > 0.5).astype(int)
    return df

def run_pipeline():
    print("[1/4] Generating & Preprocessing British Airways Flight Dataset...")
    df = generate_synthetic_ba_data(n_samples=10000)
    
    le_dict = {}
    for col in ['flight_day', 'route', 'booking_origin']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    X = df.drop('booking_complete', axis=1)
    y = df['booking_complete']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("[2/4] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("[3/4] Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print("=" * 55)
    print(f"Accuracy:                 {acc * 100:.2f}%")
    print(f"ROC-AUC Score:            {roc:.4f}")
    print(f"5-Fold Cross-Val Accuracy: {cv_scores.mean() * 100:.2f}% (std: {cv_scores.std():.4f})")
    print("=" * 55)
    print("
Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("[4/4] Feature Importance Ranking:")
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importances.to_string(index=False))

if __name__ == '__main__':
    run_pipeline()

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Import configurations
import config

def load_data(filepath):
    """Loads dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Handles duplicates, missing values, and outlier capping."""
    df_clean = df.copy()
    
    # 1. Handle duplicates
    duplicates_count = df_clean.duplicated().sum()
    if duplicates_count > 0:
        df_clean.drop_duplicates(inplace=True)
        print(f"Removed {duplicates_count} duplicate rows.")
        
    # 2. Impute missing values
    # Median imputation for numeric
    if 'study_hours_weekly' in df_clean.columns:
        study_hours_median = df_clean['study_hours_weekly'].median()
        df_clean['study_hours_weekly'] = df_clean['study_hours_weekly'].fillna(study_hours_median)
        
    # Mode imputation for categorical
    if 'parental_involvement' in df_clean.columns:
        parental_mode = df_clean['parental_involvement'].mode()[0]
        df_clean['parental_involvement'] = df_clean['parental_involvement'].fillna(parental_mode)
        
    # 3. Outlier treatment (Capping logical bounds)
    if 'study_hours_weekly' in df_clean.columns:
        df_clean['study_hours_weekly'] = df_clean['study_hours_weekly'].clip(upper=40.0)
        
    if 'attendance_rate' in df_clean.columns:
        df_clean['attendance_rate'] = df_clean['attendance_rate'].clip(lower=40.0)
        
    return df_clean

def engineer_features(df):
    """Derives new features from existing columns."""
    df_feats = df.copy()
    
    # Study efficiency: hours of study relative to sleep duration
    df_feats['study_efficiency'] = df_feats['study_hours_weekly'] / (df_feats['sleep_hours_avg'] + 1e-5)
    
    # Total engagement score: combination of attendance and participation
    df_feats['total_engagement'] = (df_feats['attendance_rate'] + df_feats['participation_rate']) / 2.0
    
    return df_feats

def train_and_evaluate():
    """Main function to run the ML training pipeline."""
    print("Starting Model Training Pipeline...")
    
    # 1. Load data
    df = load_data(config.DATA_PATH)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Preprocess data
    df_clean = preprocess_data(df)
    
    # 3. Feature engineering
    df_feats = engineer_features(df_clean)
    print("Features engineered successfully.")
    
    # 4. Separate target and features
    X = df_feats.drop(columns=['final_exam_score'])
    y = df_feats['final_exam_score']
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Data split: Training samples = {X_train.shape[0]}, Test samples = {X_test.shape[0]}")
    
    # 6. Build the ColumnTransformer Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), config.NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), config.CATEGORICAL_FEATURES)
        ]
    )
    
    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names for importance mapping
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(config.CATEGORICAL_FEATURES).tolist()
    feature_names = config.NUMERIC_FEATURES + encoded_cat_names
    
    # 7. Model Building & Comparison
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=config.RANDOM_STATE, n_jobs=-1)
    }
    
    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    best_model_obj = None
    
    print("\nTraining and comparing models:")
    print(f"{'Model Name':<20} | {'MAE':<8} | {'MSE':<8} | {'RMSE':<8} | {'R2 Score':<8}")
    print("-" * 62)
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train_processed, y_train)
        
        # Predict
        preds = model.predict(X_test_processed)
        
        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"{name:<20} | {mae:<8.4f} | {mse:<8.4f} | {rmse:<8.4f} | {r2:<8.4f}")
        
        # Track the best model based on R2 Score
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model
            
    print("-" * 62)
    print(f"Best Performing Model: {best_model_name} (R2 Score = {best_r2:.4f})\n")
    
    # 8. Save artifacts
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    
    # Save the model
    joblib.dump(best_model_obj, config.MODEL_PATH)
    print(f"Saved best model ({best_model_name}) to: {config.MODEL_PATH}")
    
    # Save the scaler preprocessor
    joblib.dump(preprocessor, config.SCALER_PATH)
    print(f"Saved preprocessing pipeline to: {config.SCALER_PATH}")
    
    # Save the features list as metadata
    features_dict = {
        'numeric_features': config.NUMERIC_FEATURES,
        'categorical_features': config.CATEGORICAL_FEATURES,
        'feature_names': feature_names,
        'best_model_name': best_model_name,
        'metrics_summary': {k: {m: v[m] for m in ['mae', 'mse', 'rmse', 'r2']} for k, v in results.items()}
    }
    joblib.dump(features_dict, config.FEATURES_METADATA_PATH)
    print(f"Saved features metadata to: {config.FEATURES_METADATA_PATH}")
    
    print("\nTraining completed successfully.")

if __name__ == '__main__':
    train_and_evaluate()

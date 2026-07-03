import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import config

@st.cache_resource
def load_ml_artifacts():
    """Loads and caches the model, preprocessor, and features metadata."""
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        return None, None, None
        
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    features_metadata = joblib.load(config.FEATURES_METADATA_PATH)
    
    return model, scaler, features_metadata

@st.cache_data
def load_cohort_data():
    """Loads and caches the historical student performance data for EDA."""
    if not os.path.exists(config.DATA_PATH):
        return None
    return pd.read_csv(config.DATA_PATH)

def inject_custom_css():
    """Reads and injects styles.css into the Streamlit application page."""
    if os.path.exists(config.STYLES_CSS_PATH):
        with open(config.STYLES_CSS_PATH, 'r') as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS stylesheet not found.")

def predict_performance(input_data, model, preprocessor):
    """Processes raw user inputs, creates derived features, and returns predicted performance score."""
    # Convert input dict to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Re-apply feature engineering
    # 1. Study efficiency: study_hours / sleep_hours
    df_input['study_efficiency'] = df_input['study_hours_weekly'] / (df_input['sleep_hours_avg'] + 1e-5)
    
    # 2. Total engagement: average of attendance and participation
    df_input['total_engagement'] = (df_input['attendance_rate'] + df_input['participation_rate']) / 2.0
    
    # Ensure columns match training schema and order
    # Preprocessor requires all features in correct order
    # Columns: ['attendance_rate', 'study_hours_weekly', 'participation_rate', 'assignment_avg_score', 'midterm_score', 'sleep_hours_avg', 'extracurricular_hours_weekly', 'parental_involvement', 'internet_access']
    # The preprocessor will scale numeric_features and encode categorical_features
    
    # Transform using fitted preprocessor ColumnTransformer
    X_processed = preprocessor.transform(df_input)
    
    # Run prediction
    prediction = model.predict(X_processed)[0]
    
    # Clip to physical limits [0, 100]
    prediction = np.clip(prediction, 0.0, 100.0)
    
    return float(prediction)

def render_metric_card(title, value, description=None, delta=None, border_color=None):
    """Renders a beautifully styled HTML/CSS metric card."""
    border_style = f"border-top: 4px solid {border_color};" if border_color else ""
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
    desc_html = f"<div class='metric-desc'>{description}</div>" if description else ""
    
    html = f"""
    <div class="custom-card" style="{border_style}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {desc_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

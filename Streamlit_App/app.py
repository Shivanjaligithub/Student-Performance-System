import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Add project root and Streamlit_App to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils

# 1. Page Configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse sidebar since it's a single page
)

# 2. Inject CSS
utils.inject_custom_css()

# 3. Header Section (Premium Center Alignment)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 25px;">
    <h1 style="margin: 0; font-size: 2.8rem;">🎓 {config.APP_TITLE}</h1>
    <p style="margin: 5px 0 0 0; color: #a5b4fc; font-size: 1.1rem; font-weight: 500;">{config.SUB_TITLE}</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# 4. Load Model Artifacts
model, scaler, metadata = utils.load_ml_artifacts()

if model is None or scaler is None or metadata is None:
    st.error("⚠️ Model Artifacts Not Found!")
    st.warning("Please run the training pipeline first to build and save the model.")
    
    # Inline trigger to train model
    if st.button("🚀 Run Training Pipeline Now"):
        with st.spinner("Training models, generating artifacts... Please wait..."):
            try:
                # Add Model directory to path to import train
                sys.path.append(os.path.join(config.BASE_DIR, 'Model'))
                import train
                train.train_and_evaluate()
                st.success("🎉 Models successfully trained and artifacts saved! Please refresh the page.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to run training script: {e}")
else:
    # Model exists, extract metadata
    best_model_name = metadata.get('best_model_name', 'Trained Model')
    
    # Two Column Layout: Inputs on Left, Results/Visuals on Right
    col_inputs, col_results = st.columns([5, 4], gap="large")
    
    with col_inputs:
        st.markdown("### 📝 Student Academic & Engagement Profile")
        st.caption("Adjust the parameters below to configure the student's metrics:")
        
        # Unified Form
        with st.form("academic_prediction_form"):
            
            # Sub-section 1: Academic Grades
            st.markdown("##### 📊 Term Grades")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                midterm = st.slider(
                    "Midterm Exam Score", 
                    min_value=30.0, 
                    max_value=100.0, 
                    value=75.0, 
                    step=1.0,
                    help="Midterm exam grade percentage."
                )
            with col_g2:
                assignment_avg = st.slider(
                    "Assignment Avg Score", 
                    min_value=40.0, 
                    max_value=100.0, 
                    value=75.0, 
                    step=1.0,
                    help="Weighted average of all coursework assignments."
                )
                
            st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
            
            # Sub-section 2: Attendance & Engagement
            st.markdown("##### 🎒 Attendance & Engagement")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                attendance = st.slider(
                    "Attendance Rate (%)", 
                    min_value=40.0, 
                    max_value=100.0, 
                    value=85.0, 
                    step=1.0,
                    help="Percentage of scheduled classes attended."
                )
            with col_e2:
                participation = st.slider(
                    "Class Participation Rate (%)", 
                    min_value=30.0, 
                    max_value=100.0, 
                    value=70.0, 
                    step=1.0,
                    help="Classroom active involvement score."
                )
                
            st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
            
            # Sub-section 3: Study Habits & Sleep
            st.markdown("##### ⏱️ Workload & Lifestyle")
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                study_hours = st.slider(
                    "Weekly Study Hours", 
                    min_value=0.0, 
                    max_value=40.0, 
                    value=15.0, 
                    step=0.5,
                    help="Average self-reported weekly study hours."
                )
            with col_l2:
                sleep_hours = st.slider(
                    "Average Daily Sleep (Hours)", 
                    min_value=4.0, 
                    max_value=10.0, 
                    value=7.0, 
                    step=0.5,
                    help="Average hours of sleep per night."
                )
                
            st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
            
            # Sub-section 4: Environment & Other Details
            st.markdown("##### 🏠 Context Factors")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                extracurricular = st.slider(
                    "Extracurricular Hours", 
                    min_value=0.0, 
                    max_value=20.0, 
                    value=6.0, 
                    step=0.5,
                    help="Hours spent on extracurricular activities."
                )
            with col_c2:
                parental = st.selectbox(
                    "Parental Involvement", 
                    options=['Low', 'Medium', 'High'],
                    index=1,
                    help="Assessment of parental support."
                )
            with col_c3:
                internet = st.selectbox(
                    "Internet Access at Home", 
                    options=['Yes', 'No'],
                    index=0,
                    help="Does the student have reliable home internet?"
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔮 Calculate Predicted Final Score")
            
    with col_results:
        st.markdown("### 📊 Performance Inference")
        
        if submit_btn:
            # Package inputs
            input_dict = {
                'attendance_rate': attendance,
                'study_hours_weekly': study_hours,
                'participation_rate': participation,
                'assignment_avg_score': assignment_avg,
                'midterm_score': midterm,
                'sleep_hours_avg': sleep_hours,
                'extracurricular_hours_weekly': extracurricular,
                'parental_involvement': parental,
                'internet_access': internet
            }
            
            # Predict
            with st.spinner("Processing through ML pipeline..."):
                pred_score = utils.predict_performance(input_dict, model, scaler)
                
            # Classify results and set style classes
            if pred_score >= 85.0:
                card_class = "predict-high"
                category = "Outstanding Performance (Grade A)"
                feedback = "Excellent results predicted! The student exhibits strong academic and study patterns. No immediate interventions needed."
                border_color = config.COLOR_SECONDARY
            elif pred_score >= 70.0:
                card_class = "predict-med"
                category = "Satisfactory Performance (Grade B/C)"
                feedback = "Steady score expected. Regular review of attendance and exam preps can boost performance further."
                border_color = config.COLOR_PRIMARY
            elif pred_score >= 50.0:
                card_class = "predict-low"
                category = "Risk of Academic Underperformance (Grade D/E)"
                feedback = "Borderline score. Consider increasing study hours weekly, raising attendance, or suggesting tutoring."
                border_color = config.COLOR_WARNING
            else:
                card_class = "predict-critical"
                category = "Critical Support Required (Grade F)"
                feedback = "⚠️ Action Needed! The student is at high risk of failing the course. Immediate academic counselor intervention is recommended."
                border_color = config.COLOR_DANGER
                
            # Render prediction card
            st.markdown(f"""
            <div class="prediction-box {card_class}">
                <div class="prediction-label">Predicted Final Score</div>
                <div class="prediction-score">{pred_score:.1f}%</div>
                <div class="prediction-category">{category}</div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: {pred_score}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback block
            st.markdown(f"""
            <div class="info-section" style="border-left-color: {border_color};">
                <h4 style="margin-top: 0;">Recommendation & Context:</h4>
                <p style="margin: 0; font-size: 0.95rem;">{feedback}</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # Welcome/Instructions placeholder box
            st.markdown("""
            <div class="custom-card" style="border-top: 4px solid #6366f1; text-align: center; padding: 40px 20px !important;">
                <h4 style="margin: 0 0 10px 0;">Predictor Ready</h4>
                <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                    Configure the student's metrics on the left form and click <b>Calculate Predicted Final Score</b> to run the machine learning model.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Feature importance / driver chart
        st.markdown("### 📊 What factors drive this prediction?")
        
        feature_names = metadata.get('feature_names', None)
        has_importance = hasattr(model, 'feature_importances_')
        has_coef = hasattr(model, 'coef_')
        
        if feature_names and (has_importance or has_coef):
            if has_importance:
                importances = model.feature_importances_
                title_suffix = "(Feature Importance)"
            else:
                # Coefficients for Linear Regression
                importances = np.abs(model.coef_)
                if np.sum(importances) > 0:
                    importances = importances / np.sum(importances)
                title_suffix = "(Relative Coefficient Weight)"
                
            indices = np.argsort(importances)[::-1]
            
            sorted_feats = [feature_names[i].replace('_', ' ').title() for i in indices]
            sorted_importances = importances[indices]
            
            fig_importance = px.bar(
                x=sorted_importances[:7], # Top 7 features
                y=sorted_feats[:7],
                orientation='h',
                color=sorted_importances[:7],
                color_continuous_scale="Purples",
                labels={'x': 'Relative Weight', 'y': 'Feature'},
                title=f"Top Driver Attributes {title_suffix}"
            )
            fig_importance.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Outfit",
                font_color="#cbd5e1",
                height=300,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            fig_importance.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.caption("Driver analysis is only available for models exposing coefficients or feature importances.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>© 2026 {config.ORGANIZATION} | Powered by ML Engine: {best_model_name if 'best_model_name' in locals() else ''}</div>", unsafe_allow_html=True)

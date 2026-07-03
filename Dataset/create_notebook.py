import json
import os

def build_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_markdown(source_lines):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines]
        })

    def add_code(source_lines):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_lines]
        })

    # --- CELL 1: Title ---
    add_markdown([
        "# Academic Performance Analysis & Machine Learning Lifecycle",
        "### Project: Performance Insight Dashboard (ML Capstone)",
        "---",
        "This notebook covers the complete ML lifecycle to predict student final exam scores based on demographic and academic performance metrics. We will proceed through the following phases:",
        "1. **Problem Understanding & Definition**",
        "2. **Data Collection**",
        "3. **Data Preprocessing** (duplicates, missing values, outliers, encoding, scaling)",
        "4. **Exploratory Data Analysis (EDA)** (univariate, bivariate, correlation)",
        "5. **Feature Engineering** (derived features, selection)",
        "6. **Model Building** (Linear Regression, Random Forest, XGBoost)",
        "7. **Model Evaluation & Selection**",
        "8. **Model Deployment Prep** (exporting artifacts)"
    ])

    # --- CELL 2: Problem Understanding ---
    add_markdown([
        "## 1. Problem Understanding",
        "**Objective**: Predict a student's final exam score (range 0-100) based on their study habits, attendance, and academic indicators. This is a regression task.",
        "### Feature Dictionary:",
        "- `attendance_rate`: Percentage of classes attended by the student (0% to 100%).",
        "- `study_hours_weekly`: Self-reported weekly study hours.",
        "- `participation_rate`: Percentage of active participation in class activities.",
        "- `assignment_avg_score`: Average score on assignments completed throughout the term.",
        "- `midterm_score`: Score achieved in the midterm exam.",
        "- `sleep_hours_avg`: Average daily hours of sleep.",
        "- `extracurricular_hours_weekly`: Weekly hours spent on extracurricular activities.",
        "- `parental_involvement`: Parental involvement level ('Low', 'Medium', 'High').",
        "- `internet_access`: Access to the internet ('Yes', 'No').",
        "- `final_exam_score` (**Target**): Final exam score (0% to 100%)."
    ])

    # --- CELL 3: Data Collection ---
    add_markdown([
        "## 2. Data Collection",
        "We load the generated synthetic dataset which mimics real-world academic data."
    ])

    add_code([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import plotly.express as px",
        "import plotly.graph_objects as go",
        "import os",
        "import joblib",
        "",
        "# Set style",
        "sns.set_theme(style='whitegrid')",
        "plt.rcParams['figure.figsize'] = (10, 6)",
        "plt.rcParams['font.size'] = 12"
    ])

    add_code([
        "# Load dataset",
        "dataset_path = '../Dataset/student_performance_raw.csv'",
        "df = pd.read_csv(dataset_path)",
        "print(f'Dataset Shape: {df.shape}')",
        "df.head()"
    ])

    # --- CELL 4: Data Preprocessing ---
    add_markdown([
        "## 3. Data Preprocessing",
        "In this phase, we address three major issues:",
        "1. **Duplicate Records**",
        "2. **Missing Values**",
        "3. **Outliers**",
        "4. **Categorical Encoding**",
        "5. **Feature Scaling**"
    ])

    add_code([
        "# 3.1 Duplicate Records Check",
        "duplicates = df.duplicated().sum()",
        "print(f'Duplicate records found: {duplicates}')",
        "",
        "# Drop duplicates",
        "if duplicates > 0:",
        "    df.drop_duplicates(inplace=True)",
        "    print(f'Shape after removing duplicates: {df.shape}')"
    ])

    add_code([
        "# 3.2 Missing Values Analysis",
        "print('Missing Values per feature:')",
        "print(df.isnull().sum())",
        "",
        "# Impute numeric features with median",
        "study_hours_median = df['study_hours_weekly'].median()",
        "df['study_hours_weekly'] = df['study_hours_weekly'].fillna(study_hours_median)",
        "",
        "# Impute categorical features with mode",
        "parental_mode = df['parental_involvement'].mode()[0]",
        "df['parental_involvement'] = df['parental_involvement'].fillna(parental_mode)",
        "",
        "print('\\nMissing values after imputation:')",
        "print(df.isnull().sum())"
    ])

    add_code([
        "# 3.3 Outlier Detection & Treatment",
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))",
        "sns.boxplot(ax=axes[0], data=df, y='study_hours_weekly', color='#3b82f6')",
        "axes[0].set_title('Boxplot of Study Hours Weekly (Before Capping)')",
        "sns.boxplot(ax=axes[1], data=df, y='attendance_rate', color='#10b981')",
        "axes[1].set_title('Boxplot of Attendance Rate (Before Capping)')",
        "plt.show()",
        "",
        "# Outlier treatment: Capping using logical bounds",
        "# study_hours_weekly: cap at 40 hours",
        "df['study_hours_weekly'] = df['study_hours_weekly'].clip(upper=40.0)",
        "",
        "# attendance_rate: cap minimum at 40.0% (as lower values represent extreme outliers)",
        "df['attendance_rate'] = df['attendance_rate'].clip(lower=40.0)",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))",
        "sns.boxplot(ax=axes[0], data=df, y='study_hours_weekly', color='#60a5fa')",
        "axes[0].set_title('Boxplot of Study Hours Weekly (After Capping)')",
        "sns.boxplot(ax=axes[1], data=df, y='attendance_rate', color='#34d399')",
        "axes[1].set_title('Boxplot of Attendance Rate (After Capping)')",
        "plt.show()"
    ])

    # --- CELL 5: EDA ---
    add_markdown([
        "## 4. Exploratory Data Analysis (EDA)",
        "We perform univariate, bivariate, and correlation analysis to uncover relationships between variables."
    ])

    add_code([
        "# 4.1 Univariate Analysis: Distribution of Target (Final Exam Score)",
        "sns.histplot(data=df, x='final_exam_score', kde=True, color='#8b5cf6', bins=30)",
        "plt.title('Distribution of Final Exam Score')",
        "plt.xlabel('Final Exam Score')",
        "plt.ylabel('Count')",
        "plt.show()"
    ])

    add_code([
        "# 4.2 Bivariate Analysis: Numerical features vs Target",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
        "",
        "sns.scatterplot(ax=axes[0,0], data=df, x='midterm_score', y='final_exam_score', alpha=0.6, color='#ef4444')",
        "axes[0,0].set_title('Midterm Score vs Final Exam Score')",
        "",
        "sns.scatterplot(ax=axes[0,1], data=df, x='assignment_avg_score', y='final_exam_score', alpha=0.6, color='#f59e0b')",
        "axes[0,1].set_title('Assignment Average Score vs Final Exam Score')",
        "",
        "sns.scatterplot(ax=axes[1,0], data=df, x='study_hours_weekly', y='final_exam_score', alpha=0.6, color='#3b82f6')",
        "axes[1,0].set_title('Weekly Study Hours vs Final Exam Score')",
        "",
        "sns.scatterplot(ax=axes[1,1], data=df, x='attendance_rate', y='final_exam_score', alpha=0.6, color='#10b981')",
        "axes[1,1].set_title('Attendance Rate vs Final Exam Score')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    add_code([
        "# 4.3 Bivariate Analysis: Categorical Features vs Target",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))",
        "",
        "sns.boxplot(ax=axes[0], data=df, x='parental_involvement', y='final_exam_score', order=['Low', 'Medium', 'High'], palette='Purples')",
        "axes[0].set_title('Parental Involvement vs Final Exam Score')",
        "",
        "sns.boxplot(ax=axes[1], data=df, x='internet_access', y='final_exam_score', palette='Set2')",
        "axes[1].set_title('Internet Access vs Final Exam Score')",
        "",
        "plt.show()"
    ])

    add_code([
        "# 4.4 Correlation Analysis",
        "numeric_cols = df.select_dtypes(include=[np.number]).columns",
        "corr_matrix = df[numeric_cols].corr()",
        "",
        "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)",
        "plt.title('Correlation Matrix of Academic Features')",
        "plt.show()"
    ])

    # --- CELL 6: Feature Engineering ---
    add_markdown([
        "## 5. Feature Engineering",
        "We create two derived features to enrich the dataset:",
        "1. `study_efficiency`: Ratio of study hours to sleep hours (how efficiently a student studies given sleep duration).",
        "2. `total_engagement`: Average of attendance rate and participation rate."
    ])

    add_code([
        "# 5.1 Create Derived Features",
        "df['study_efficiency'] = df['study_hours_weekly'] / (df['sleep_hours_avg'] + 1e-5)",
        "df['total_engagement'] = (df['attendance_rate'] + df['participation_rate']) / 2.0",
        "",
        "print('New features added. Previews of dataset:')",
        "df[['study_hours_weekly', 'sleep_hours_avg', 'study_efficiency', 'attendance_rate', 'participation_rate', 'total_engagement']].head()"
    ])

    # --- CELL 7: Model Building ---
    add_markdown([
        "## 6. Model Building",
        "We will split the data into training (80%) and testing (20%) sets, then train three models:",
        "1. **Linear Regression** (Baseline)",
        "2. **Random Forest Regressor** (Non-linear Ensemble)",
        "3. **XGBoost Regressor** (Gradient Boosting)"
    ])

    add_code([
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.linear_model import LinearRegression",
        "from sklearn.ensemble import RandomForestRegressor",
        "from xgboost import XGBRegressor",
        "",
        "# Separate features and target",
        "X = df.drop(columns=['final_exam_score'])",
        "y = df['final_exam_score']",
        "",
        "# Define features list",
        "numeric_features = ['attendance_rate', 'study_hours_weekly', 'participation_rate', ",
        "                    'assignment_avg_score', 'midterm_score', 'sleep_hours_avg', ",
        "                    'extracurricular_hours_weekly', 'study_efficiency', 'total_engagement']",
        "categorical_features = ['parental_involvement', 'internet_access']",
        "",
        "# Train/Test Split",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
        "print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')"
    ])

    add_code([
        "# 6.2 Building the Preprocessing Pipeline",
        "# We use standard scaling for numeric features, and one-hot encoding for categorical features",
        "preprocessor = ColumnTransformer(",
        "    transformers=[",
        "        ('num', StandardScaler(), numeric_features),",
        "        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)",
        "    ]",
        ")",
        "",
        "# Fit the preprocessor on training data",
        "X_train_processed = preprocessor.fit_transform(X_train)",
        "X_test_processed = preprocessor.transform(X_test)",
        "print('Preprocessing pipeline successfully built and fitted.')"
    ])

    add_code([
        "# 6.3 Train Models",
        "models = {",
        "    'Linear Regression': LinearRegression(),",
        "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),",
        "    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)",
        "}",
        "",
        "trained_models = {}",
        "predictions = {}",
        "",
        "for name, model in models.items():",
        "    print(f'Training {name}...')",
        "    model.fit(X_train_processed, y_train)",
        "    trained_models[name] = model",
        "    predictions[name] = model.predict(X_test_processed)",
        "print('All models trained successfully.')"
    ])

    # --- CELL 8: Model Evaluation ---
    add_markdown([
        "## 7. Model Evaluation",
        "We evaluate the model performance using:",
        "- **Mean Absolute Error (MAE)**",
        "- **Mean Squared Error (MSE)**",
        "- **Root Mean Squared Error (RMSE)**",
        "- **Coefficient of Determination ($R^2$ Score)**"
    ])

    add_code([
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        "",
        "metrics_summary = []",
        "",
        "for name, preds in predictions.items():",
        "    mae = mean_absolute_error(y_test, preds)",
        "    mse = mean_squared_error(y_test, preds)",
        "    rmse = np.sqrt(mse)",
        "    r2 = r2_score(y_test, preds)",
        "    ",
        "    metrics_summary.append({",
        "        'Model': name,",
        "        'MAE': mae,",
        "        'MSE': mse,",
        "        'RMSE': rmse,",
        "        'R2 Score': r2",
        "    })",
        "",
        "metrics_df = pd.DataFrame(metrics_summary)",
        "print('Comparison Table of Model Performance:')",
        "metrics_df"
    ])

    add_code([
        "# Plotting Actual vs Predicted",
        "fig, axes = plt.subplots(1, 3, figsize=(20, 6))",
        "for i, (name, preds) in enumerate(predictions.items()):",
        "    sns.scatterplot(ax=axes[i], x=y_test, y=preds, alpha=0.5, color='#4f46e5')",
        "    # Perfect prediction line",
        "    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)",
        "    axes[i].set_title(f'{name}\\n(R2: {metrics_df.loc[metrics_df[\"Model\"]==name, \"R2 Score\"].values[0]:.4f})')",
        "    axes[i].set_xlabel('Actual Final Score')",
        "    axes[i].set_ylabel('Predicted Final Score')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    add_code([
        "# Plot Random Forest Feature Importance to verify drivers",
        "rf_model = trained_models['Random Forest']",
        "# Retrieve feature names from preprocessing",
        "cat_encoder = preprocessor.named_transformers_['cat']",
        "encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()",
        "feature_names = numeric_features + encoded_cat_names",
        "",
        "importances = rf_model.feature_importances_",
        "indices = np.argsort(importances)[::-1]",
        "",
        "plt.figure(figsize=(12, 6))",
        "sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')",
        "plt.title('Feature Importances (Random Forest)')",
        "plt.xlabel('Relative Importance')",
        "plt.ylabel('Feature')",
        "plt.show()"
    ])

    # --- CELL 9: Save Best Model ---
    add_markdown([
        "## 8. Exporting artifacts for deployment",
        "We save the best performing model (XGBoost or Random Forest) along with the preprocessor pipeline."
    ])

    add_code([
        "# Find the model with the highest R2 score",
        "best_model_name = metrics_df.loc[metrics_df['R2 Score'].idxmax(), 'Model']",
        "best_model = trained_models[best_model_name]",
        "print(f'Best Model is: {best_model_name}')",
        "",
        "# Save artifacts",
        "os.makedirs('../Model', exist_ok=True)",
        "model_file_path = '../Model/best_model.pkl'",
        "preprocessor_file_path = '../Model/scaler.pkl'",
        "",
        "joblib.dump(best_model, model_file_path)",
        "joblib.dump(preprocessor, preprocessor_file_path)",
        "",
        "# Save the features list as well",
        "features_dict = {",
        "    'numeric_features': numeric_features,",
        "    'categorical_features': categorical_features,",
        "    'feature_names': feature_names",
        "}",
        "joblib.dump(features_dict, '../Model/features_dict.pkl')",
        "",
        "print('Best model, Preprocessor pipeline, and features metadata saved to Model/ folder.')"
    ])

    # Save to file
    os.makedirs('Notebook', exist_ok=True)
    with open('Notebook/academic_performance_analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook/academic_performance_analysis.ipynb successfully created.")

if __name__ == '__main__':
    build_notebook()

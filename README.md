# Academic Analytics Dashboard (Machine Learning Capstone)

A production-quality machine learning capstone project designed to predict student academic performance (continuous score and risk classification) based on attendance, study habits, class engagement, and midterm exams.

The system uses Python, scikit-learn, and XGBoost for modeling, and is served via an interactive, modern multi-page Streamlit dashboard.

---

## 📂 Project Directory Structure

The project strictly follows the required folder layout:

```
Student Performance Prediction System_Shivanjali Ojha/
│
├── Dataset/
│   ├── student_performance_raw.csv    # Generated raw dataset (1000+ records)
│   ├── generate_dataset.py            # Programmatic synthetic data generator
│   └── create_notebook.py             # Script to programmatically compile the notebook
│
├── Notebook/
│   └── academic_performance_analysis.ipynb # Complete Jupyter Notebook covering the ML lifecycle
│
├── Model/
│   ├── config.py                      # Shared branding & file path configs
│   ├── train.py                       # Modular ML pipeline training script
│   ├── best_model.pkl                 # Active serialized model (Random Forest / XGBoost)
│   ├── scaler.pkl                     # Serialized scaling & encoding ColumnTransformer pipeline
│   └── features_dict.pkl              # Saved feature lists & comparative validation metrics
│
├── Streamlit_App/
│   ├── app.py                         # Application Entry Point & Cohort KPIs (Home)
│   ├── config.py                      # Global Streamlit branding & color configs
│   ├── utils.py                       # Core utility functions (caching, scaling, custom HTML cards)
│   ├── styles.css                     # Premium glassmorphic styling sheet
│   └── pages/
│       ├── 1_Data_Insights.py         # Cohort EDA visualization page (Plotly Express)
│       ├── 2_Performance_Prediction.py# Real-time inference form & dynamic driver charts
│       └── 3_About.py                 # Technical documentation & comparative metrics comparison
│
├── Documentation/
│   ├── project_report_outline.md      # Detailed Chapter 1-6 template outline for project report
│   └── presentation_outline.md        # Slide-by-slide outline for the presentation deck
│
├── requirements.txt                   # Pinned project dependencies
└── README.md                          # Setup and operations guide (This file)
```

---

## ⚡ Setup & Installation

Follow these steps to set up the project locally on your machine.

### 1. Clone or Extract the Project
Ensure the project is in a directory (e.g. `Student Performance Prediction System_Shivanjali Ojha`).

### 2. Install Dependencies
Open your command prompt or terminal in the project root folder and run:
```bash
pip install -r requirements.txt
```
*Note: The requirements file lists standard packages compatible with newer Python releases (including Python 3.13).*

### 3. Generate the Dataset & Jupyter Notebook
If they are not already compiled, you can regenerate the raw dataset and the Jupyter notebook file by executing:
```bash
# Generate the student raw CSV dataset
python Dataset/generate_dataset.py

# Programmatically generate the Jupyter Notebook
python Dataset/create_notebook.py
```

### 4. Train the ML Models & Export Artifacts
Train and evaluate the models (Linear Regression, Random Forest, and XGBoost) and save the highest-performing model automatically:
```bash
python Model/train.py
```
This script will:
- Load the raw dataset from `Dataset/`.
- Handle duplicates, impute missing values, and clip outliers.
- Engineer derived features (`study_efficiency` and `total_engagement`).
- Train and evaluate all models.
- Save the best-performing model to `Model/best_model.pkl`, the preprocessing scaler to `Model/scaler.pkl`, and the model metrics comparison to `Model/features_dict.pkl`.

### 5. Launch the Streamlit Web Application
Start the interactive dashboard locally:
```bash
streamlit run Streamlit_App/app.py
```
A browser window will open automatically at `http://localhost:8501`.

---

## 🎨 Branding Rule & Customization

The application adheres to the branding rule. There is **no hardcoded reference** to the specific project topic name ("Student Performance Prediction") in the display headers, titles, or page menus.
Instead, display names are loaded from configuration constants.

To rebrand or customize the application name or organization:
1. Open [Model/config.py](file:///c:/Users/justi/OneDrive/Desktop/Student%20Performance%20Prediction%20System_Shivanjali%20Ojha/Model/config.py) and [Streamlit_App/config.py](file:///c:/Users/justi/OneDrive/Desktop/Student%20Performance%20Prediction%20System_Shivanjali%20Ojha/Streamlit_App/config.py).
2. Edit `APP_TITLE` (e.g. `"Academic Analytics Dashboard"`), `SUB_TITLE`, and `ORGANIZATION`.
3. Save the files. The updates will propagate across all pages instantly.

---

## 🧠 Model Pipeline Summary

- **Algorithms Compared**:
  - **Linear Regression** (Explainable Baseline)
  - **Random Forest Regressor** (Robust Bagging Ensemble)
  - **XGBoost Regressor** (Gradient Boosting Trees)
- **Primary Metrics**: $R^2$ Score, MAE, MSE, RMSE.
- **Top Predictors**: Midterm score, assignment average, class attendance, and study hours weekly.
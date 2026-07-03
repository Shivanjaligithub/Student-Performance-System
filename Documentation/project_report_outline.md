# Academic Report Outline: Performance Insight & Analytics System

*This document serves as an academic report template for the Machine Learning Capstone project. Replace placeholders and instructional text in square brackets `[...]` with your specific project details.*

---

## Title Page
**Title**: Predictive Academic Modeling and Cohort Engagement Analytics  
**Course**: Machine Learning Capstone  
**Author**: Shivanjali Ojha  
**Date**: [Date]  

---

## Table of Contents
1. Chapter 1: Introduction
2. Chapter 2: Literature Review
3. Chapter 3: Methodology
4. Chapter 4: Implementation
5. Chapter 5: Results and Discussion
6. Chapter 6: Conclusion and Future Scope
7. References

---

## Chapter 1: Introduction

### 1.1 Background
*Instruction: Provide a broad introduction to the role of data science and machine learning in modern education (EdTech). Explain the shift from traditional reactive student support to proactive, predictive models.*

[Insert background text. Discuss how analytics can identify risk factors before exams, helping instructors personalize guidance.]

### 1.2 Problem Statement
*Instruction: Clearly define the problem you are solving. Focus on the cost of student dropouts, low engagement, and the difficulty of manually tracking academic progress in large cohorts.*

[Insert problem statement. E.g., "Educators often lack real-time predictive indicators of student success. Traditional grading registers progress too late to permit preventative interventions. This project develops a continuous regression model to forecast exam outcomes based on engagement indicators..."]

### 1.3 Objectives
- To generate and clean a multi-variable student dataset representing realistic academic behaviors.
- To execute exploratory data analysis (EDA) to find the primary drivers of student performance.
- To implement, tune, and compare three regression algorithms: Linear Regression, Random Forest, and XGBoost.
- To deploy a responsive, multi-page web application permitting educators to run real-time student success inference.

---

## Chapter 2: Literature Review

### 2.1 Existing Approaches to Performance Prediction
*Instruction: Review traditional academic prediction models (e.g., GPA forecasting, simple linear regressions, or rule-based warning systems). Contrast them with modern machine learning techniques.*

[Insert literature review. Discuss previous works that utilized student engagement logs, LMS (Learning Management System) interaction metrics, and demographic details.]

### 2.2 Machine Learning in EdTech
*Instruction: Document findings from 2-3 recent research papers concerning the use of Random Forest, SVMs, and Gradient Boosting in predicting student grades. Cite metrics (e.g., R², MAE) reported in the literature.*

[Insert discussion of ensemble methods and tree boosting in regression studies.]

---

## Chapter 3: Methodology

### 3.1 Dataset Description
*Instruction: Detail the variables in the dataset. Explain their types, ranges, and educational significance.*

| Feature Name | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `attendance_rate` | Numeric | 40% - 100% | Class attendance percentage (capped at 40% floor). |
| `study_hours_weekly` | Numeric | 0 - 40 hours | Weekly study time outside class (capped at 40 hours). |
| `participation_rate`| Numeric | 30% - 100% | Engagement score based on class participation. |
| `assignment_avg` | Numeric | 40 - 100 | Average marks in term assignments. |
| `midterm_score` | Numeric | 30 - 100 | Score obtained in the midterm exam. |
| `sleep_hours_avg` | Numeric | 4 - 10 hours | Average daily hours of sleep. |
| `extracurricular` | Numeric | 0 - 20 hours | Hours spent in non-academic activities. |
| `parental_involvement`| Categorical| Low, Medium, High | Level of parental support. |
| `internet_access` | Categorical| Yes, No | Availability of home internet connection. |
| `final_exam_score` (Target)| Numeric | 0 - 100 | The regression target (final grade). |

### 3.2 Data Preprocessing Workflow
1. **Deduplication**: Identification and removal of duplicate records to prevent training leakage.
2. **Imputation**: Handling missing values using median imputation for continuous features and mode imputation for categorical attributes.
3. **Outlier Treatment**: Boundary capping using IQR-inspired logical limits (e.g., clipping study hours to a max of 40 and attendance to a minimum of 40%).
4. **Encoding**: One-Hot Encoding for categorical features (`internet_access`) and Ordinal/One-Hot encoding for parental support.
5. **Scaling**: Standard scaling of numerical columns to normalize input distributions.

### 3.3 Algorithms Used
*Instruction: Briefly explain the theory behind each algorithm.*
- **Linear Regression**: Finds the line of best fit by minimizing the sum of squared residuals. Useful as an explainable baseline.
- **Random Forest Regressor**: Ensembles multiple decision trees using bagging. Reduces variance and captures non-linear decision boundaries.
- **XGBoost Regressor**: A gradient boosting library that builds trees sequentially to correct prior model residuals using regularized objective functions.

---

## Chapter 4: Implementation

### 4.1 Exploratory Data Analysis (EDA)
*Instruction: Describe key patterns discovered in your data. Refer to visual plots (histograms, scatter plots, correlation heatmap).*

[Insert EDA discussion. For example: "The correlation matrix revealed that midterm scores ($r \approx 0.72$) and assignment averages ($r \approx 0.68$) are the strongest predictors of final exam performance. Weekly study hours also showed a strong positive association ($r \approx 0.55$)."]

### 4.2 Feature Engineering & Selection
*Instruction: Detail how you created the derived features.*
- `study_efficiency`: Computed as `study_hours_weekly / (sleep_hours_avg + 1e-5)`. Captures the quality/intensity of study relative to rest.
- `total_engagement`: Computed as the average of `attendance_rate` and `participation_rate`.

### 4.3 Model Development
*Instruction: Detail your train-test split (80-20), cross-validation, and library frameworks (scikit-learn, xgboost).*

[Insert model setup details and parameters used.]

---

## Chapter 5: Results and Discussion

### 5.1 Model Evaluation Metrics
We evaluate our models using MAE, MSE, RMSE, and $R^2$ Score.
- **Mean Absolute Error (MAE)**: Measures average magnitude of errors.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily.
- **$R^2$ Score**: Quantifies the proportion of target variance explained by the features.

### 5.2 Comparative Analysis
*Instruction: Fill in the actual scores obtained by your training pipeline.*

| Model Name | MAE | MSE | RMSE | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | [Insert MAE] | [Insert MSE] | [Insert RMSE] | [Insert R²] |
| **Random Forest** | [Insert MAE] | [Insert MSE] | [Insert RMSE] | [Insert R²] |
| **XGBoost** | [Insert MAE] | [Insert MSE] | [Insert RMSE] | [Insert R²] |

### 5.3 Discussion & Interpretability
*Instruction: Discuss which model was selected, why, and feature importances. Analyze how attendance and midterm grades drive final outcomes.*

[Insert discussion text. Explain how the best model was saved as a joblib pickle file and loaded into the dashboard.]

---

## Chapter 6: Conclusion and Future Scope

### 6.1 Conclusion
[Insert conclusion summary. Highlight the successful development of the pipeline and the deployment of the Streamlit application.]

### 6.2 Future Scope
- Integration with live Learning Management Systems (LMS) like Canvas or Moodle.
- Addition of cognitive features like student stress indexes or questionnaire responses.
- Transitioning the regression model to trigger automated email alerts for students flagged "at-risk".

---

## References
1. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
3. [Insert additional academic papers or textbooks utilized during the research.]

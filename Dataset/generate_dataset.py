import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Generate core independent features
    # Attendance Rate: mean 85%, std 10%, bounded [50%, 100%]
    attendance = np.random.normal(85, 10, num_samples)
    attendance = np.clip(attendance, 50, 100)
    
    # Study Hours Weekly: mean 15 hrs, std 6 hrs, bounded [2, 40]
    study_hours = np.random.normal(15, 6, num_samples)
    study_hours = np.clip(study_hours, 2, 40)
    
    # Participation Rate: mean 70%, std 15%, bounded [30%, 100%]
    participation = np.random.normal(70, 15, num_samples)
    participation = np.clip(participation, 30, 100)
    
    # Assignment Average Score: mean 75, std 12, bounded [40, 100]
    assignment_avg = np.random.normal(75, 12, num_samples)
    assignment_avg = np.clip(assignment_avg, 40, 100)
    
    # Midterm Score: correlated with assignment average
    # midterm = 0.6 * assignment_avg + 0.4 * normal_noise, bounded [30, 100]
    midterm = 0.7 * assignment_avg + np.random.normal(20, 10, num_samples)
    midterm = np.clip(midterm, 30, 100)
    
    # Sleep Hours: mean 7.2 hrs, std 1.2 hrs, bounded [4, 10]
    sleep_hours = np.random.normal(7.2, 1.2, num_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    # Extracurricular Hours Weekly: mean 6 hrs, std 4 hrs, bounded [0, 20]
    extracurricular_hours = np.random.normal(6, 4, num_samples)
    extracurricular_hours = np.clip(extracurricular_hours, 0, 20)
    
    # Parental Involvement: categorical ('Low', 'Medium', 'High')
    parental_choices = ['Low', 'Medium', 'High']
    parental_probs = [0.2, 0.5, 0.3]
    parental_involvement = np.random.choice(parental_choices, num_samples, p=parental_probs)
    
    # Internet Access: categorical ('Yes', 'No')
    internet_choices = ['Yes', 'No']
    internet_probs = [0.85, 0.15]
    internet_access = np.random.choice(internet_choices, num_samples, p=internet_probs)
    
    # 2. Calculate continuous Target Score: Final Exam Score (0 to 100)
    # Define linear combinations with reasonable weights
    # Max points potential contributions:
    # Midterm: 0.35 * 100 = 35
    # Assignment Avg: 0.25 * 100 = 25
    # Attendance: 0.15 * 100 = 15
    # Study Hours: 0.4 * 40 = 16
    # Participation: 0.05 * 100 = 5
    # Sleep: quadratic effect, optimal at 7.5 hours (up to 4 points)
    # Parental: Low=0, Medium=2, High=4
    # Internet: No=0, Yes=2
    
    sleep_factor = 4.0 - 1.5 * (sleep_hours - 7.5)**2
    sleep_factor = np.clip(sleep_factor, 0, 4)
    
    parental_map = {'Low': 0, 'Medium': 2, 'High': 4}
    parental_factor = np.array([parental_map[val] for val in parental_involvement])
    
    internet_map = {'No': 0, 'Yes': 2}
    internet_factor = np.array([internet_map[val] for val in internet_access])
    
    # Base calculation
    final_score = (
        0.35 * midterm +
        0.25 * assignment_avg +
        0.15 * attendance +
        0.40 * study_hours +
        0.05 * participation +
        sleep_factor +
        parental_factor +
        internet_factor
    )
    
    # Add random noise (std = 3) to represent other variables/exam day variance
    noise = np.random.normal(0, 3, num_samples)
    final_score = final_score + noise
    
    # Clip final score to be strictly between 0 and 100
    final_score = np.clip(final_score, 0, 100)
    
    # Assemble into DataFrame
    df = pd.DataFrame({
        'attendance_rate': attendance,
        'study_hours_weekly': study_hours,
        'participation_rate': participation,
        'assignment_avg_score': assignment_avg,
        'midterm_score': midterm,
        'sleep_hours_avg': sleep_hours,
        'extracurricular_hours_weekly': extracurricular_hours,
        'parental_involvement': parental_involvement,
        'internet_access': internet_access,
        'final_exam_score': final_score
    })
    
    # 3. Introduce data issues for preprocessing practice:
    # A. Introduce missing values (approx 2% in study_hours_weekly and parental_involvement)
    missing_study_indices = np.random.choice(num_samples, size=int(0.02 * num_samples), replace=False)
    df.loc[missing_study_indices, 'study_hours_weekly'] = np.nan
    
    missing_parental_indices = np.random.choice(num_samples, size=int(0.02 * num_samples), replace=False)
    df.loc[missing_parental_indices, 'parental_involvement'] = np.nan
    
    # B. Introduce Outliers (specifically, set a few values to unreasonable amounts to test scaling/capping)
    # We will set 5 rows to have study_hours_weekly of 80 (extreme outlier)
    outlier_study_indices = np.random.choice(num_samples, size=5, replace=False)
    df.loc[outlier_study_indices, 'study_hours_weekly'] = 80.0
    
    # We will set 5 rows to have attendance_rate of 5% (extreme outlier)
    outlier_attendance_indices = np.random.choice(num_samples, size=5, replace=False)
    df.loc[outlier_attendance_indices, 'attendance_rate'] = 5.0
    
    # C. Introduce duplicate rows
    duplicate_rows = df.sample(n=10, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    # Ensure Dataset directory exists
    os.makedirs('Dataset', exist_ok=True)
    
    df = generate_synthetic_data()
    file_path = os.path.join('Dataset', 'student_performance_raw.csv')
    df.to_csv(file_path, index=False)
    print(f"Dataset successfully generated with {len(df)} samples (including duplicates) and saved to {file_path}")

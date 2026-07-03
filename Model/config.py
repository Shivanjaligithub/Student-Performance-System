import os

# Project Branding (Central Configurable Constant)
APP_NAME = "Academic Analytics Dashboard"
ORGANIZATION = "EduInsight Analytics Group"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Dataset', 'student_performance_raw.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Model', 'scaler.pkl')
FEATURES_METADATA_PATH = os.path.join(BASE_DIR, 'Model', 'features_dict.pkl')

# Train configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features lists
NUMERIC_FEATURES = [
    'attendance_rate', 
    'study_hours_weekly', 
    'participation_rate',
    'assignment_avg_score', 
    'midterm_score', 
    'sleep_hours_avg',
    'extracurricular_hours_weekly', 
    'study_efficiency', 
    'total_engagement'
]

CATEGORICAL_FEATURES = ['parental_involvement', 'internet_access']

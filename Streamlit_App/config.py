import os

# Project branding configuration (Adhering to Branding Rule)
# Modify these to rebrand the entire application instantly
APP_TITLE = "Academic Analytics Dashboard"
SUB_TITLE = "Performance Insight & Analytics Platform"
ORGANIZATION = "EduInsight Analytics Group"

# Theme colors for visualizations and branding
COLOR_PRIMARY = "#6366f1"   # Indigo
COLOR_SECONDARY = "#10b981" # Emerald
COLOR_WARNING = "#f59e0b"   # Amber
COLOR_DANGER = "#ef4444"    # Rose
COLOR_BACKGROUND = "#f8fafc" # Light slate background (standard)
COLOR_TEXT = "#0f172a"      # Dark slate text

# Paths (using absolute paths to resolve reliably from anywhere)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Model', 'scaler.pkl')
FEATURES_METADATA_PATH = os.path.join(BASE_DIR, 'Model', 'features_dict.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Dataset', 'student_performance_raw.csv')
STYLES_CSS_PATH = os.path.join(BASE_DIR, 'Streamlit_App', 'styles.css')

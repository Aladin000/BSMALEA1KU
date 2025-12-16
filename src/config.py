from pathlib import Path

# Reproducibility
SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Column names
ID_COL = 'IDpol'  
TARGET_COL = 'ClaimFrequency'  # ClaimNb / Exposure
CLAIM_NB_COL = 'ClaimNb'
EXPOSURE_COL = 'Exposure'

# Feature categories by encoding type
ORDINAL_FEATURES = ['Area']  # Features with natural order
NOMINAL_FEATURES = ['VehBrand', 'VehGas', 'Region']  # Features without natural order
CATEGORICAL_FEATURES = ORDINAL_FEATURES + NOMINAL_FEATURES  # All categorical features
NUMERICAL_FEATURES = ['VehPower', 'VehAge', 'DrivAge', 'Density', 'BonusMalus']
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Ordinal encoding order (Area: A=most rural to F=most urban)
AREA_ORDER = ['A', 'B', 'C', 'D', 'E', 'F']

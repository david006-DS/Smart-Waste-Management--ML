"""
Configuration for Waste Management ML Pipeline
================================================
Centralized settings for features, models, and deployment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Feature sets for different prediction tasks."""
    
    # Fill Level Prediction Features
    fill_level_features: List[str] = field(default_factory=lambda: [
        # Temporal
        'days_since_last_collection',
        'month',
        'is_weekend',
        'is_holiday',
        'is_festival_period',
        # Bin characteristics
        'capacity_liters',
        'nearby_population',
        'has_lid',
        # Location (encoded)
        'location_type',
        'road_accessibility',
        # Historical
        'fill_rate_7day_avg',
        'prev_fill_level',
        # Environmental
        'temperature_c',
        'rainfall_mm',
    ])
    
    # Waste Type Classification Features
    waste_type_features: List[str] = field(default_factory=lambda: [
        'location_type',
        'district',
        'capacity_liters',
        'fill_level_percent',
        'fill_rate_per_day',
        'waste_weight_kg',
        'is_festival_period',
        'month',
        'nearby_population',
    ])
    
    # Priority Scoring Features
    priority_features: List[str] = field(default_factory=lambda: [
        'fill_level_percent',
        'waste_type_primary',  # Encoded
        'days_since_last_collection',
        'nearby_population',
        'distance_to_depot_km',
        'road_accessibility',
        'overflow_reported',
        'odor_complaint',
    ])
    
    # Categorical features requiring encoding
    categorical_features: List[str] = field(default_factory=lambda: [
        'location_type',
        'district',
        'road_accessibility',
        'waste_type_primary',
        'day_of_week',
        'bin_material',
    ])
    
    # Target columns
    fill_level_target: str = 'fill_level_category'
    waste_type_target: str = 'waste_type_primary'


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Model hyperparameters and training settings."""
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Train/test split
    test_size: float = 0.2
    
    # Cross-validation
    cv_folds: int = 5
    
    # Decision Tree baseline
    dt_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'class_weight': 'balanced',
        'random_state': 42,
    })
    
    # Random Forest
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42,
    })
    
    # XGBoost
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 8,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
    })
    
    # Hyperparameter search spaces
    rf_search_space: Dict[str, List] = field(default_factory=lambda: {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [3, 5, 10],
    })
    
    xgb_search_space: Dict[str, List] = field(default_factory=lambda: {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 8, 12],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [3, 5, 10],
    })


# =============================================================================
# CLASS MAPPINGS
# =============================================================================

FILL_LEVEL_CLASSES = ['low', 'medium', 'high', 'critical']
FILL_LEVEL_MAP = {label: idx for idx, label in enumerate(FILL_LEVEL_CLASSES)}
FILL_LEVEL_MAP_INV = {idx: label for idx, label in enumerate(FILL_LEVEL_CLASSES)}

WASTE_TYPE_CLASSES = ['organic', 'recyclable', 'general', 'hazardous', 'e_waste', 'construction']
WASTE_TYPE_MAP = {label: idx for idx, label in enumerate(WASTE_TYPE_CLASSES)}
WASTE_TYPE_MAP_INV = {idx: label for idx, label in enumerate(WASTE_TYPE_CLASSES)}

LOCATION_TYPES = ['residential', 'commercial', 'market', 'industrial', 'institutional', 'hospitality']
ROAD_ACCESSIBILITY = ['easy', 'moderate', 'difficult']

# Priority weights for safety-critical classes
HAZARDOUS_PRIORITY_WEIGHT = 3.0  # Hazardous waste gets 3x priority
CRITICAL_FILL_WEIGHT = 2.0  # Critical fill level gets 2x priority


# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

@dataclass
class DeploymentConfig:
    """Settings for model deployment."""
    
    # Model versioning
    model_version: str = "1.0.0"
    
    # Confidence thresholds
    min_confidence: float = 0.6  # Minimum confidence to report prediction
    hazardous_confidence: float = 0.4  # Lower threshold for safety-critical
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Streamlit settings
    streamlit_port: int = 8501


# =============================================================================
# INSTANTIATE DEFAULT CONFIGS
# =============================================================================

FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()
DEPLOYMENT_CONFIG = DeploymentConfig()

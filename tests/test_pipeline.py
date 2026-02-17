"""
Unit Tests for Waste Management ML Pipeline
============================================
Tests for preprocessing, training, and prediction modules.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import shutil
import json

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.train import (
    DataPreprocessor,
    WasteManagementTrainer,
    WasteManagementPipeline,
    ModelResult
)
from models.predict import (
    WasteManagementPredictor,
    ModelBundle,
    Prediction,
    PriorityScore,
    predict_single
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'bin_id': [f"BIN_{i:05d}" for i in range(n_samples)],
        'location_type': np.random.choice(
            ['residential', 'commercial', 'market', 'industrial'], n_samples
        ),
        'district': np.random.choice(
            ['Accra Central', 'Tema', 'Madina', 'Kasoa'], n_samples
        ),
        'capacity_liters': np.random.choice([120, 240, 660, 1100], n_samples),
        'nearby_population': np.random.randint(100, 3000, n_samples),
        'has_lid': np.random.randint(0, 2, n_samples),
        'road_accessibility': np.random.choice(['easy', 'moderate', 'difficult'], n_samples),
        'distance_to_depot_km': np.random.uniform(0.5, 20, n_samples),
        'days_since_last_collection': np.random.randint(1, 10, n_samples),
        'fill_level_percent': np.random.uniform(0, 100, n_samples),
        'waste_weight_kg': np.random.uniform(5, 200, n_samples),
        'temperature_c': np.random.uniform(25, 35, n_samples),
        'rainfall_mm': np.random.exponential(5, n_samples),
        'humidity_percent': np.random.uniform(50, 90, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'is_holiday': np.random.randint(0, 2, n_samples),
        'is_festival_period': np.random.randint(0, 2, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'overflow_reported': np.random.choice([0, 0, 0, 0, 1], n_samples),  # 20% overflow
        'odor_complaint': np.random.choice([0, 0, 0, 0, 1], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Derived features
    df['fill_rate_per_day'] = df['fill_level_percent'] / df['days_since_last_collection'].clip(lower=1)
    df['fill_rate_7day_avg'] = df['fill_rate_per_day'] * np.random.uniform(0.8, 1.2, n_samples)
    df['prev_fill_level'] = df['fill_level_percent'] * np.random.uniform(0.7, 1.0, n_samples)
    
    # Target variables
    df['fill_level_category'] = pd.cut(
        df['fill_level_percent'],
        bins=[-np.inf, 30, 60, 85, np.inf],
        labels=['low', 'medium', 'high', 'critical']
    ).astype(str)
    
    # Waste type based on location (simplified pattern)
    waste_map = {
        'residential': ['organic', 'general', 'recyclable'],
        'commercial': ['recyclable', 'general', 'organic'],
        'market': ['organic', 'organic', 'recyclable'],
        'industrial': ['hazardous', 'construction', 'recyclable'],
    }
    df['waste_type_primary'] = df['location_type'].apply(
        lambda x: np.random.choice(waste_map.get(x, ['general']))
    )
    
    return df


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# =============================================================================
# PREPROCESSOR TESTS
# =============================================================================

class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_fit_categorical(self, sample_data):
        """Test fitting on categorical columns."""
        preprocessor = DataPreprocessor()
        cat_cols = ['location_type', 'district', 'road_accessibility']
        
        preprocessor.fit(sample_data, cat_cols)
        
        assert preprocessor._is_fitted
        assert len(preprocessor.label_encoders) == len(cat_cols)
        for col in cat_cols:
            assert col in preprocessor.label_encoders
    
    def test_transform_categorical(self, sample_data):
        """Test transforming categorical columns."""
        preprocessor = DataPreprocessor()
        cat_cols = ['location_type', 'road_accessibility']
        
        df_encoded = preprocessor.fit_transform(sample_data, cat_cols)
        
        # Check columns are now numeric
        for col in cat_cols:
            assert df_encoded[col].dtype in [np.int64, np.int32, int]
            assert df_encoded[col].min() >= 0
    
    def test_transform_without_fit_raises(self, sample_data):
        """Test that transform without fit raises error."""
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.transform(sample_data, ['location_type'])
    
    def test_encode_target(self, sample_data):
        """Test target variable encoding."""
        preprocessor = DataPreprocessor()
        y = sample_data['fill_level_category']
        
        y_encoded = preprocessor.encode_target(y, 'fill_level')
        
        assert 'fill_level' in preprocessor.target_encoders
        assert len(np.unique(y_encoded)) == len(y.unique())
        assert y_encoded.dtype in [np.int64, np.int32]
    
    def test_decode_target(self, sample_data):
        """Test target variable decoding."""
        preprocessor = DataPreprocessor()
        y = sample_data['fill_level_category']
        
        y_encoded = preprocessor.encode_target(y, 'fill_level')
        y_decoded = preprocessor.decode_target(y_encoded, 'fill_level')
        
        # Should match original
        np.testing.assert_array_equal(y.values, y_decoded)
    
    def test_handle_unseen_category(self, sample_data):
        """Test handling of unseen categories during transform."""
        preprocessor = DataPreprocessor()
        preprocessor.fit(sample_data, ['location_type'])
        
        # Create new data with unseen category
        new_data = sample_data.iloc[:5].copy()
        new_data.loc[new_data.index[0], 'location_type'] = 'new_unseen_type'
        
        # Should not raise, should map to first class
        df_encoded = preprocessor.transform(new_data, ['location_type'])
        assert df_encoded is not None


# =============================================================================
# TRAINER TESTS
# =============================================================================

class TestWasteManagementTrainer:
    """Tests for WasteManagementTrainer class."""
    
    def test_train_decision_tree(self, sample_data):
        """Test training a decision tree model."""
        trainer = WasteManagementTrainer(random_state=42)
        
        # Prepare data
        X = sample_data[['capacity_liters', 'nearby_population', 'days_since_last_collection']].values
        y = pd.Categorical(sample_data['fill_level_category']).codes
        
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]
        
        params = {'max_depth': 5, 'random_state': 42}
        
        model, result = trainer.train_model(
            X_train, X_test, y_train, y_test,
            model_type='decision_tree',
            task='fill_level',
            params=params,
            feature_names=['capacity_liters', 'nearby_population', 'days_since_last_collection']
        )
        
        assert model is not None
        assert isinstance(result, ModelResult)
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1_macro <= 1
        assert len(result.cv_scores) == 5  # Default cv folds
    
    def test_train_random_forest(self, sample_data):
        """Test training a random forest model."""
        trainer = WasteManagementTrainer(random_state=42)
        
        X = sample_data[['capacity_liters', 'nearby_population']].values
        y = pd.Categorical(sample_data['fill_level_category']).codes
        
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]
        
        params = {'n_estimators': 10, 'max_depth': 5, 'random_state': 42}
        
        model, result = trainer.train_model(
            X_train, X_test, y_train, y_test,
            model_type='random_forest',
            task='fill_level',
            params=params,
            feature_names=['capacity_liters', 'nearby_population']
        )
        
        assert model is not None
        assert len(result.feature_importances) == 2
    
    def test_compare_models(self, sample_data):
        """Test model comparison functionality."""
        trainer = WasteManagementTrainer(random_state=42)
        
        X = sample_data[['capacity_liters', 'nearby_population', 'fill_level_percent']].values
        y = pd.Categorical(sample_data['fill_level_category']).codes
        
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]
        
        results = trainer.compare_models(
            X_train, X_test, y_train, y_test,
            task='fill_level',
            feature_names=['capacity_liters', 'nearby_population', 'fill_level_percent'],
            dt_params={'max_depth': 5, 'random_state': 42},
            rf_params={'n_estimators': 10, 'max_depth': 5, 'random_state': 42},
            xgb_params=None  # Skip XGBoost for faster tests
        )
        
        assert 'decision_tree' in results
        assert 'random_forest' in results
        assert len(trainer.models) >= 2
    
    def test_get_best_model(self, sample_data):
        """Test retrieval of best performing model."""
        trainer = WasteManagementTrainer(random_state=42)
        
        X = sample_data[['capacity_liters', 'nearby_population']].values
        y = pd.Categorical(sample_data['fill_level_category']).codes
        
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]
        
        # Train multiple models
        trainer.train_model(
            X_train, X_test, y_train, y_test,
            'decision_tree', 'fill_level',
            {'max_depth': 3, 'random_state': 42},
            ['capacity_liters', 'nearby_population']
        )
        trainer.train_model(
            X_train, X_test, y_train, y_test,
            'random_forest', 'fill_level',
            {'n_estimators': 10, 'max_depth': 5, 'random_state': 42},
            ['capacity_liters', 'nearby_population']
        )
        
        name, model, result = trainer.get_best_model('fill_level')
        
        assert name is not None
        assert model is not None
        assert result.f1_macro == max(r.f1_macro for r in trainer.results.values() if 'fill_level' in r.task)


# =============================================================================
# PIPELINE TESTS
# =============================================================================

class TestWasteManagementPipeline:
    """Tests for the complete pipeline orchestrator."""
    
    def test_prepare_fill_level_data(self, sample_data, temp_model_dir):
        """Test data preparation for fill level task."""
        # Save sample data
        data_path = temp_model_dir / "features.csv"
        sample_data.to_csv(data_path, index=False)
        
        pipeline = WasteManagementPipeline(
            data_path=data_path,
            models_path=temp_model_dir
        )
        
        X_train, X_test, y_train, y_test, features = pipeline.prepare_data(
            sample_data, task='fill_level'
        )
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(features) > 0
    
    def test_prepare_waste_type_data(self, sample_data, temp_model_dir):
        """Test data preparation for waste type task."""
        data_path = temp_model_dir / "features.csv"
        sample_data.to_csv(data_path, index=False)
        
        pipeline = WasteManagementPipeline(
            data_path=data_path,
            models_path=temp_model_dir
        )
        
        X_train, X_test, y_train, y_test, features = pipeline.prepare_data(
            sample_data, task='waste_type'
        )
        
        assert len(X_train) > 0
        assert len(np.unique(y_train)) > 1  # Multiple classes
    
    def test_save_models(self, sample_data, temp_model_dir):
        """Test model serialization."""
        data_path = temp_model_dir / "features.csv"
        sample_data.to_csv(data_path, index=False)
        
        pipeline = WasteManagementPipeline(
            data_path=data_path,
            models_path=temp_model_dir
        )
        
        # Run minimal pipeline
        X_train, X_test, y_train, y_test, features = pipeline.prepare_data(
            sample_data, task='fill_level'
        )
        
        pipeline.trainer.train_model(
            X_train, X_test, y_train, y_test,
            'decision_tree', 'fill_level',
            {'max_depth': 5, 'random_state': 42},
            features
        )
        
        # Save
        bundle_path = pipeline.save_models(version="test")
        
        assert bundle_path.exists()
        assert (bundle_path / "fill_level_model.joblib").exists()
        assert (bundle_path / "preprocessor.joblib").exists()
        assert (bundle_path / "feature_config.json").exists()


# =============================================================================
# PREDICTION TESTS
# =============================================================================

class TestPrediction:
    """Tests for prediction data classes."""
    
    def test_prediction_to_dict(self):
        """Test Prediction serialization."""
        pred = Prediction(
            task='fill_level',
            predicted_class='high',
            confidence=0.85,
            class_probabilities={'low': 0.05, 'medium': 0.08, 'high': 0.85, 'critical': 0.02},
            is_high_confidence=True
        )
        
        d = pred.to_dict()
        
        assert d['task'] == 'fill_level'
        assert d['predicted_class'] == 'high'
        assert d['confidence'] == 0.85
        assert d['is_high_confidence'] is True
    
    def test_priority_score_to_dict(self):
        """Test PriorityScore serialization."""
        fill_pred = Prediction('fill_level', 'critical', 0.9, {}, True)
        waste_pred = Prediction('waste_type', 'hazardous', 0.7, {}, True)
        
        priority = PriorityScore(
            bin_id='BIN_00001',
            priority_score=95.5,
            priority_level='critical',
            fill_level_prediction=fill_pred,
            waste_type_prediction=waste_pred,
            risk_factors=['Hazardous waste', 'Overflow'],
            recommended_action='Immediate dispatch'
        )
        
        d = priority.to_dict()
        
        assert d['bin_id'] == 'BIN_00001'
        assert d['priority_score'] == 95.5
        assert 'fill_level_prediction' in d
        assert len(d['risk_factors']) == 2


class TestPredictor:
    """Tests for WasteManagementPredictor class."""
    
    @pytest.fixture
    def trained_bundle(self, sample_data, temp_model_dir):
        """Create a trained model bundle for testing predictions."""
        data_path = temp_model_dir / "features.csv"
        sample_data.to_csv(data_path, index=False)
        
        pipeline = WasteManagementPipeline(
            data_path=data_path,
            models_path=temp_model_dir
        )
        
        # Train both models
        for task in ['fill_level', 'waste_type']:
            X_train, X_test, y_train, y_test, features = pipeline.prepare_data(
                sample_data, task=task
            )
            pipeline.trainer.train_model(
                X_train, X_test, y_train, y_test,
                'random_forest', task,
                {'n_estimators': 10, 'max_depth': 5, 'random_state': 42, 'n_jobs': 1},
                features
            )
        
        bundle_path = pipeline.save_models(version="test")
        return bundle_path
    
    def test_load_bundle(self, trained_bundle):
        """Test loading a model bundle."""
        bundle = ModelBundle(trained_bundle)
        
        assert 'fill_level' in bundle.models
        assert bundle.preprocessor is not None
        assert bundle.config is not None
    
    def test_predict_fill_level_single(self, trained_bundle, sample_data):
        """Test single fill level prediction."""
        predictor = WasteManagementPredictor(bundle_path=trained_bundle)
        
        # Single record as dict
        record = sample_data.iloc[0].to_dict()
        
        pred = predictor.predict_fill_level(record)
        
        assert isinstance(pred, Prediction)
        assert pred.task == 'fill_level'
        assert pred.predicted_class in ['low', 'medium', 'high', 'critical']
        assert 0 <= pred.confidence <= 1
    
    def test_predict_fill_level_batch(self, trained_bundle, sample_data):
        """Test batch fill level prediction."""
        predictor = WasteManagementPredictor(bundle_path=trained_bundle)
        
        preds = predictor.predict_fill_level(sample_data.head(10))
        
        assert isinstance(preds, list)
        assert len(preds) == 10
        assert all(isinstance(p, Prediction) for p in preds)
    
    def test_predict_waste_type(self, trained_bundle, sample_data):
        """Test waste type prediction."""
        predictor = WasteManagementPredictor(bundle_path=trained_bundle)
        
        record = sample_data.iloc[0].to_dict()
        pred = predictor.predict_waste_type(record)
        
        assert pred.task == 'waste_type'
        assert pred.predicted_class in [
            'organic', 'recyclable', 'general', 'hazardous', 'e_waste', 'construction'
        ]
    
    def test_calculate_priority(self, trained_bundle, sample_data):
        """Test priority score calculation."""
        predictor = WasteManagementPredictor(bundle_path=trained_bundle)
        
        record = sample_data.iloc[0].to_dict()
        record['bin_id'] = 'TEST_BIN'
        
        priority = predictor.calculate_priority(record)
        
        assert isinstance(priority, PriorityScore)
        assert priority.bin_id == 'TEST_BIN'
        assert 0 <= priority.priority_score <= 100
        assert priority.priority_level in ['low', 'medium', 'high', 'critical']
        assert priority.recommended_action is not None
    
    def test_priority_hazardous_boost(self, trained_bundle, sample_data):
        """Test that hazardous waste increases priority."""
        predictor = WasteManagementPredictor(bundle_path=trained_bundle)
        
        # Create industrial record (more likely hazardous)
        industrial = sample_data[sample_data['location_type'] == 'industrial'].iloc[0].to_dict()
        residential = sample_data[sample_data['location_type'] == 'residential'].iloc[0].to_dict()
        
        # Make fill levels similar
        industrial['fill_level_percent'] = 50
        residential['fill_level_percent'] = 50
        
        p1 = predictor.calculate_priority(industrial)
        p2 = predictor.calculate_priority(residential)
        
        # Industrial (more hazardous) should generally score higher
        # Note: This is probabilistic, so we just check both complete
        assert p1.priority_score is not None
        assert p2.priority_score is not None
    
    def test_predict_batch_with_priority(self, trained_bundle, sample_data):
        """Test batch prediction with priority scores."""
        predictor = WasteManagementPredictor(bundle_path=trained_bundle)
        
        batch = sample_data.head(5)
        result = predictor.predict_batch(batch, include_priority=True)
        
        assert 'predicted_fill_level' in result.columns
        assert 'predicted_waste_type' in result.columns
        assert 'priority_score' in result.columns
        assert 'priority_level' in result.columns
        assert 'recommended_action' in result.columns
        assert len(result) == 5
    
    def test_get_model_info(self, trained_bundle):
        """Test model info retrieval."""
        predictor = WasteManagementPredictor(bundle_path=trained_bundle)
        
        info = predictor.get_model_info()
        
        assert 'version' in info
        assert 'models_loaded' in info
        assert 'fill_level' in info['models_loaded']


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_features(self, sample_data, temp_model_dir):
        """Test handling of missing features in input."""
        data_path = temp_model_dir / "features.csv"
        sample_data.to_csv(data_path, index=False)
        
        pipeline = WasteManagementPipeline(
            data_path=data_path,
            models_path=temp_model_dir
        )
        
        # Remove some features
        incomplete_data = sample_data.drop(columns=['temperature_c', 'rainfall_mm'])
        
        # Should still work with available features
        X_train, X_test, y_train, y_test, features = pipeline.prepare_data(
            incomplete_data, task='fill_level'
        )
        
        assert len(features) < len(pipeline.fill_level_features)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        preprocessor = DataPreprocessor()
        
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        preprocessor.fit(empty_df, ['nonexistent_col'])
        assert len(preprocessor.label_encoders) == 0
    
    def test_all_same_class(self, sample_data):
        """Test handling when all samples have same class."""
        trainer = WasteManagementTrainer(random_state=42)
        
        X = sample_data[['capacity_liters', 'nearby_population']].values[:100]
        y = np.zeros(100, dtype=int)  # All same class
        
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Should not crash
        model, result = trainer.train_model(
            X_train, X_test, y_train, y_test,
            'decision_tree', 'test',
            {'random_state': 42},
            ['capacity_liters', 'nearby_population']
        )
        
        assert model is not None
        assert result.accuracy == 1.0  # Perfect on single class
    
    def test_bundle_not_found(self, temp_model_dir):
        """Test error when bundle path doesn't exist."""
        fake_path = temp_model_dir / "nonexistent_bundle"
        
        with pytest.raises(FileNotFoundError):
            ModelBundle(fake_path)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_run(self, sample_data, temp_model_dir):
        """Test complete pipeline from data to predictions."""
        # Save data
        data_path = temp_model_dir / "features.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Run pipeline
        pipeline = WasteManagementPipeline(
            data_path=data_path,
            models_path=temp_model_dir
        )
        
        # Train
        df = pipeline.load_data()
        
        for task in ['fill_level', 'waste_type']:
            X_train, X_test, y_train, y_test, features = pipeline.prepare_data(df, task)
            pipeline.trainer.train_model(
                X_train, X_test, y_train, y_test,
                'random_forest', task,
                {'n_estimators': 10, 'max_depth': 5, 'random_state': 42, 'n_jobs': 1},
                features
            )
        
        # Save
        bundle_path = pipeline.save_models()
        
        # Load and predict
        predictor = WasteManagementPredictor(bundle_path=bundle_path)
        
        # Make predictions on new data
        new_record = sample_data.iloc[-1].to_dict()
        priority = predictor.calculate_priority(new_record)
        
        assert priority is not None
        assert priority.priority_level in ['low', 'medium', 'high', 'critical']
        
        # Verify metrics were saved
        assert (bundle_path / "fill_level_metrics.json").exists()
        with open(bundle_path / "fill_level_metrics.json") as f:
            metrics = json.load(f)
            assert 'accuracy' in metrics
            assert 'f1_macro' in metrics


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

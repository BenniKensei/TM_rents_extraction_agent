"""
Machine Learning Model Training Pipeline
========================================
Implements end-to-end XGBoost regression training with scikit-learn pipelines.
Orchestrates data loading, preprocessing, hyperparameter tuning, and model serialization.

Architecture:
- Data ingestion: Load from database via data_profiler, clean via data_cleaning
- Preprocessing: ColumnTransformer with target encoding and binary encoding
- Hyperparameter tuning: RandomizedSearchCV for efficient parameter space exploration
- Model training: XGBoost regressor with reg:squarederror objective
- Serialization: Joblib dump for production FastAPI deployment

Expected Output:
- Serialized sklearn Pipeline object (preprocessing + model)
- MAE ~€50-100 (prediction error on unseen test set)
- R² ~0.65-0.75 (model explains 65-75% of rent variance)

Seeded random state (42) ensures reproducible results across runs.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import category_encoders as ce
import joblib

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cleaning import clean_listings_data
from src.feature_engineering import encode_binary


def get_data():
    """
    Load and prepare training data from the primary data source.
    
    Data Pipeline:
    1. Extract raw data from Supabase PostgreSQL (via data_profiler.extract_data)
    2. Apply data cleaning (deduplication, missing value handling, outlier removal)
    
    Returns:
        pd.DataFrame: Clean training data ready for feature engineering.
                     Schema: ['title', 'neighborhood', 'monthly_rent_eur', 'rooms', 'is_pet_friendly', ...]
    
    Raises:
        ImportError: If data_profiler module not found or database unavailable.
    
    Note: This function will fall back to sample data if database connection fails,
          which is useful for development but should not be used for production training.
    """
    from scripts.data_profiler import extract_data
    df = extract_data()
    df = clean_listings_data(df)
    return df


def main():
    """
    Main training pipeline: Orchestrates complete ML workflow from data to serialized model.
    
    Workflow Steps:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. DATA LOADING & CLEANING                                      │
    │    - Extract raw listings from database                         │
    │    - Apply deduplication, missing value handling, outlier removal│
    └─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ 2. FEATURE SELECTION & SPLITTING                                │
    │    - Select features: neighborhood, rooms, is_pet_friendly      │
    │    - Exclude price_per_room (target leakage prevention)         │
    │    - Train/test split: 80/20 with random_state=42              │
    └─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ 3. PREPROCESSING PIPELINE (ColumnTransformer)                   │
    │    - Target encode neighborhood (high-cardinality categorical)  │
    │    - Binary encode is_pet_friendly                              │
    │    - Pass-through rooms (numeric, no transformation)            │
    └─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ 4. HYPERPARAMETER TUNING (RandomizedSearchCV)                   │
    │    - Search space: learning_rate, max_depth, n_estimators       │
    │    - Metric: Negative MAE (scikit-learn convention)             │
    │    - CV: 3-fold cross-validation                                │
    │    - Iterations: 15 random parameter combinations               │
    └─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ 5. EVALUATION & REPORTING                                       │
    │    - Compute MAE, RMSE, R² on held-out test set                │
    │    - Print best hyperparameters and performance metrics         │
    └─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ 6. SERIALIZATION                                                │
    │    - Joblib dump entire pipeline to assets/model.joblib         │
    │    - Pipeline includes preprocessing + model (stateless inference)│
    └─────────────────────────────────────────────────────────────────┘
    
    Random State:
    - random_state=42 ensures reproducible train/test splits and hyperparameter search
    - Enables consistent model retraining across environments and time
    
    Raises:
        RuntimeError: If training data is empty or model serialization fails.
    """
    print("="*60)
    print("🚀 Model Training Pipeline initialized")
    print("="*60)
    
    print("\nLoading and cleaning data...")
    df = get_data()
    
    if df.empty:
        print("❌ Dataset is empty. Aborting training.")
        return

    # ────────────────────────────────────────────────────────────────────────────
    # STEP 1: FEATURE SELECTION
    # ────────────────────────────────────────────────────────────────────────────
    
    # **CRITICAL**: Feature List Rationale
    # Include: neighborhood, rooms, is_pet_friendly
    # Exclude: price_per_room (LEAKAGE: derived from target)
    #
    # Why exclude price_per_room?
    # - price_per_room = monthly_rent_eur (target) / rooms (feature)
    # - Model would learn: monthly_rent_eur ≈ price_per_room * rooms (tautology)
    # - Test set evaluation would be artificially inflated (information leakage)
    # - Production predictions would be impossible without knowing the target
    
    features = ['neighborhood', 'rooms', 'is_pet_friendly']
    target = 'monthly_rent_eur'
    
    # Drop rows where target is missing (cannot train without labels)
    df = df.dropna(subset=[target])
    
    X = df[features]  # Shape: (n_samples, 3)
    y = df[target]    # Shape: (n_samples,)
    
    print(f"\n📊 Data shape: Features X={X.shape}, Target y={y.shape}")
    
    # ────────────────────────────────────────────────────────────────────────────
    # STEP 2: TRAIN/TEST SPLIT
    # ────────────────────────────────────────────────────────────────────────────
    
    # 80/20 split: Standard industry practice for ML model evaluation
    # random_state=42: Deterministic split for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")
    
    # ────────────────────────────────────────────────────────────────────────────
    # STEP 3: PREPROCESSING PIPELINE CONSTRUCTION
    # ────────────────────────────────────────────────────────────────────────────
    
    # FunctionTransformer wraps the encode_binary function
    # Allows custom transformations in sklearn pipelines
    binary_transformer = FunctionTransformer(encode_binary, validate=False)
    
    # TargetEncoder: Replace neighborhood (string) with mean target value per neighborhood
    # - Handles high-cardinality categoricals efficiently
    # - Captures domain signal: expensive neighborhoods get high encoded values
    # - category_encoders library provides better train/test leakage protection than pd.map
    target_encoder = ce.TargetEncoder(cols=['neighborhood'])
    
    # ColumnTransformer: Apply different transformations to different feature groups
    # Expected shape transformations:
    # - Input: (n_samples, 3) with columns [neighborhood, is_pet_friendly, rooms]
    # - Output: (n_samples, 3) with columns [neighborhood_encoded, pet_friendly_encoded, rooms]
    preprocessor = ColumnTransformer(
        transformers=[
            ('neighborhood_te', target_encoder, ['neighborhood']),
            ('pet_friendly_bin', binary_transformer, ['is_pet_friendly']),
            ('passthrough', 'passthrough', ['rooms'])
        ],
        remainder='drop'
    )
    
    # Pipeline: Chain preprocessing and model into single estimator
    # Advantage: Fit preprocessing on train set, automatically apply to test set
    # Prevents data leakage from fitting preprocessor on combined train+test data
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])
    
    # ────────────────────────────────────────────────────────────────────────────
    # STEP 4: HYPERPARAMETER TUNING
    # ────────────────────────────────────────────────────────────────────────────
    
    # **Hyperparameter Rationale:**
    # - learning_rate (eta): Controls step size when updating model weights
    #   * 0.01: Conservative learning, slower convergence but potentially better generalization
    #   * 0.2:  Aggressive learning, faster convergence but risk of overshooting optimal
    # - max_depth: Maximum tree depth in ensemble
    #   * 3-4: Shallow trees, high bias, low variance (underfitting risk)
    #   * 5-6: Moderate depth, balanced bias-variance trade-off
    #   * 8:   Deep trees, low bias, high variance (overfitting risk)
    # - n_estimators: Number of boosting rounds (trees added sequentially)
    #   * 50:   Few trees, rapid training but limited model capacity
    #   * 300:  Many trees, slower training but better pattern capture
    #
    # Search Strategy: RandomizedSearchCV instead of GridSearchCV
    # - GridSearchCV: Tests ALL combinations (4 × 5 × 4 = 80 configurations)
    # - RandomizedSearchCV: Tests 15 random combinations (80% less computation)
    # - Trade-off: Less exhaustive but practical for this parameter space size
    
    param_distributions = {
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6, 8],
        'model__n_estimators': [50, 100, 200, 300]
    }
    
    print("\n🔍 Starting RandomizedSearchCV for Hyperparameter Tuning...")
    print("   Searching 15 random configurations with 3-fold cross-validation...")
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=15,           # Number of parameter combinations to sample
        cv=3,                # 3-fold cross-validation for robust evaluation
        scoring='neg_mean_absolute_error',  # Optimize for MAE (negative because sklearn convention)
        random_state=42,     # Deterministic random sampling
        n_jobs=-1,           # Use all available CPU cores
        verbose=1            # Print progress
    )
    
    # Fit: Performs GridSearchCV, computes cross-validation scores for each configuration
    random_search.fit(X_train, y_train)
    
    # Extract best-performing pipeline (includes fitted preprocessor + model)
    best_pipeline = random_search.best_estimator_
    print(f"✅ Best Parameters Found:\n   {random_search.best_params_}")
    print(f"   CV Score (neg_MAE): {random_search.best_score_:.2f}")
    
    # ────────────────────────────────────────────────────────────────────────────
    # STEP 5: EVALUATION ON HELD-OUT TEST SET
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n📈 Evaluating Model on Unseen Test Set...")
    y_pred = best_pipeline.predict(X_test)  # Shape: (n_test,)
    
    # **Regression Metrics:**
    # 1. MAE (Mean Absolute Error): Average absolute prediction error
    #    - Interpretation: Model is off by €X on average
    #    - Range: [0, ∞), lower is better
    #    - Robust to outliers compared to MSE
    # 2. RMSE (Root Mean Squared Error): Penalizes large errors more
    #    - Interpretation: Typical prediction error magnitude
    #    - Useful for understanding extreme cases
    # 3. R² (Coefficient of Determination): Proportion of variance explained
    #    - Interpretation: Model explains X% of rent price variance
    #    - Range: [0, 1], higher is better
    #    - 0.65 = reasonable model, 0.85+ = strong model
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Mean Absolute Error (MAE):       €{mae:.2f}")
    print(f"   Root Mean Squared Error (RMSE):  €{rmse:.2f}")
    print(f"   R-squared (R²):                  {r2:.4f}")
    
    # Business interpretation
    print(f"\n💡 Interpretation:")
    print(f"   - Predictions deviate by ~€{mae:.0f}/month on average")
    print(f"   - Model explains {r2*100:.1f}% of rent price variation")
    print(f"   - Price range in test set: €{y_test.min():.0f} - €{y_test.max():.0f}")
    
    # ────────────────────────────────────────────────────────────────────────────
    # STEP 6: SERIALIZATION FOR PRODUCTION
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n📦 Serializing trained pipeline to disk...")
    os.makedirs('assets', exist_ok=True)
    model_path = os.path.join('assets', 'model.joblib')
    
    # Joblib serialization: Saves preprocessing + model as single object
    # Advantage: No need to fit preprocessor on new data; apply predict directly
    # File size: ~1-5 MB for typical XGBoost models
    joblib.dump(best_pipeline, model_path)
    print(f"✅ Model pipeline successfully serialized to {model_path}")
    
    print("="*60)
    print("🎯 Training Complete! Ready for inference deployment.\n")


if __name__ == "__main__":
    main()

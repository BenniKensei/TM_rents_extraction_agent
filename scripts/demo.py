"""
Demonstration Script: Minimal ML Pipeline with Toy Data
========================================================

This script demonstrates the complete ML training pipeline without requiring:
- External database connections
- Web scraping setup
- LLM inference (Ollama)

Perfect for:
- Learning the codebase structure
- Validating installation (all dependencies installed correctly)
- Rapid prototyping on local machine
- CI/CD testing environments

Execution:
    $ python scripts/demo.py

Expected runtime: ~30-60 seconds (includes hyperparameter tuning)
Output: assets/model.joblib (serialized sklearn pipeline, 1.2 MB)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import clean_listings_data
from src.feature_engineering import engineer_features
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import category_encoders as ce
import joblib


def generate_sample_data(n_samples: int = 200) -> pd.DataFrame:
    """
    Generate synthetic apartment listing data for demonstration.
    
    **Rationale for Synthetic Data:**
    - Avoids database dependency (works offline)
    - Deterministic output (reproducible results)
    - Realistic distributions matching actual market
    - Includes edge cases (missing values, outliers) for testing
    
    Args:
        n_samples: Number of synthetic listings to generate (default 200).
    
    Returns:
        DataFrame with columns: ['neighborhood', 'rooms', 'is_pet_friendly', 'monthly_rent_eur']
    
    **Data Generation Strategy:**
    1. Neighborhood distribution (weighted):
       - Centru: 35% (expensive, €450-800)
       - Complexul Studentesc: 30% (moderate, €350-600)
       - Periphery: 20% (cheap, €250-450)
       - Unknown: 15% (missing values, €300-500)
    
    2. Rooms distribution (Poisson-like):
       - 1 bedroom: 40%
       - 2 bedrooms: 45%
       - 3+ bedrooms: 15%
    
    3. Pet-friendly distribution:
       - True: 25%
       - False: 75%
    
    4. Rent prices:
       - Base price: neighborhood-dependent (€300-800)
       - Room adjustment: +€150 per additional room
       - Pet premium: +€25 if pet-friendly
       - Random noise: N(0, €50) to simulate market variation
    """
    np.random.seed(42)  # Reproducibility
    
    # ─── NEIGHBORHOOD GENERATION ───
    neighborhoods = np.random.choice(
        ['Centru', 'Complexul Studentesc', 'Periphery', 'Nord', 'Sud'],
        size=n_samples,
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    # ─── ROOMS GENERATION ───
    # Most apartments are 1-2 bedrooms (realistic market distribution)
    rooms = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.4, 0.45, 0.12, 0.03])
    
    # ─── PET-FRIENDLY GENERATION ───
    is_pet_friendly = np.random.choice(
        [True, False],
        size=n_samples,
        p=[0.25, 0.75]  # Only 25% of apartments accept pets
    )
    
    # ─── RENT PRICE GENERATION ───
    # Deterministic pricing model with noise
    base_prices = {
        'Centru': 600,
        'Complexul Studentesc': 450,
        'Nord': 380,
        'Sud': 400,
        'Periphery': 320
    }
    
    monthly_rent_eur = []
    for neighborhood, room_count, pet_friendly in zip(neighborhoods, rooms, is_pet_friendly):
        base = base_prices.get(neighborhood, 400)
        room_adjustment = (room_count - 1) * 150  # €150 per additional room
        pet_premium = 25 if pet_friendly else 0
        noise = np.random.normal(0, 50)  # €50 standard deviation
        
        rent = int(base + room_adjustment + pet_premium + noise)
        # Clip to realistic range [€200, €1500]
        rent = max(200, min(1500, rent))
        monthly_rent_eur.append(rent)
    
    # ─── CONSTRUCT DATAFRAME ───
    df = pd.DataFrame({
        'title': [f'Apartment {i}' for i in range(n_samples)],
        'neighborhood': neighborhoods,
        'rooms': rooms,
        'is_pet_friendly': is_pet_friendly,
        'monthly_rent_eur': monthly_rent_eur
    })
    
    # ─── ADD SOME MISSING VALUES (5%) ───
    # Realistic data quality issue
    missing_idx = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_idx, 'neighborhood'] = None
    
    return df


def main():
    """
    Execute complete ML training pipeline on synthetic data.
    
    **Pipeline Steps:**
    1. Generate synthetic data (200 listings)
    2. Data cleaning (deduplication, missing values, outliers)
    3. Feature engineering (target encoding, binary encoding)
    4. Train/test split (80/20)
    5. Hyperparameter tuning (RandomizedSearchCV)
    6. Model evaluation (MAE, RMSE, R²)
    7. Model serialization (Joblib)
    """
    print("="*60)
    print("🚀 Demo: ML Pipeline Training (Toy Data)")
    print("="*60)
    
    # ─── STEP 1: GENERATE DATA ───
    print("\n📊 Generating synthetic apartment listing data...")
    df = generate_sample_data(n_samples=200)
    print(f"✅ Generated {len(df)} synthetic listings")
    print(f"   Neighborhoods: {df['neighborhood'].nunique()} unique areas")
    print(f"   Rooms: min={df['rooms'].min()}, max={df['rooms'].max()}, avg={df['rooms'].mean():.1f}")
    print(f"   Rent: €{df['monthly_rent_eur'].min()}-€{df['monthly_rent_eur'].max()}, avg=€{df['monthly_rent_eur'].mean():.0f}")
    
    # ─── STEP 2: DATA CLEANING ───
    print("\n🧹 Cleaning data...")
    df = clean_listings_data(df)
    print(f"✅ After cleaning: {len(df)} listings remain")
    
    if df.empty:
        print("❌ Dataset is empty after cleaning. Aborting.")
        return
    
    # ─── STEP 3: FEATURE ENGINEERING ───
    print("\n🔧 Engineering features...")
    df_engineered = engineer_features(df.copy())
    print(f"✅ Feature engineering complete")
    print(f"   Features: {df_engineered.columns.tolist()}")
    
    # ─── STEP 4: FEATURE SELECTION & SPLITTING ───
    print("\n📊 Preparing training data...")
    features = ['neighborhood', 'rooms', 'is_pet_friendly']
    target = 'monthly_rent_eur'
    
    # Drop missing targets
    df = df.dropna(subset=[target])
    
    X = df[features]
    y = df[target]
    
    print(f"   Features X shape: {X.shape}")
    print(f"   Target y shape: {y.shape}")
    
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # ─── STEP 5: BUILD PREPROCESSING PIPELINE ───
    print("\n🔨 Building sklearn pipeline...")
    
    from src.feature_engineering import encode_binary
    
    binary_transformer = FunctionTransformer(encode_binary, validate=False)
    target_encoder = ce.TargetEncoder(cols=['neighborhood'])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('neighborhood_te', target_encoder, ['neighborhood']),
            ('pet_friendly_bin', binary_transformer, ['is_pet_friendly']),
            ('passthrough', 'passthrough', ['rooms'])
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])
    
    # ─── STEP 6: HYPERPARAMETER TUNING ───
    print("\n🔍 Hyperparameter tuning (RandomizedSearchCV)...")
    
    param_distributions = {
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6, 8],
        'model__n_estimators': [50, 100, 200, 300]
    }
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_pipeline = random_search.best_estimator_
    print(f"✅ Best hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # ─── STEP 7: MODEL EVALUATION ───
    print("\n📈 Evaluating on test set...")
    y_pred = best_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Mean Absolute Error (MAE):       €{mae:.2f}")
    print(f"   Root Mean Squared Error (RMSE):  €{rmse:.2f}")
    print(f"   R-squared (R²):                  {r2:.4f}")
    
    print(f"\n💡 Interpretation:")
    print(f"   - Predictions deviate by ~€{mae:.0f}/month on average")
    print(f"   - Model explains {r2*100:.1f}% of rent price variation")
    print(f"   - Price range in test set: €{y_test.min():.0f} - €{y_test.max():.0f}")
    
    # ─── STEP 8: SERIALIZATION ───
    print("\n📦 Serializing model...")
    os.makedirs('assets', exist_ok=True)
    model_path = os.path.join('assets', 'model.joblib')
    joblib.dump(best_pipeline, model_path)
    print(f"✅ Model serialized to {model_path}")
    
    # ─── INFERENCE DEMO ───
    print("\n🎯 Testing inference on sample data...")
    sample_listings = pd.DataFrame({
        'neighborhood': ['Centru', 'Periphery'],
        'rooms': [2, 1],
        'is_pet_friendly': [True, False]
    })
    
    predictions = best_pipeline.predict(sample_listings)
    
    for i, (idx, row) in enumerate(sample_listings.iterrows()):
        print(f"   {row['neighborhood']}, {row['rooms']} room(s), pet-friendly={row['is_pet_friendly']}")
        print(f"   → Predicted rent: €{predictions[i]:.2f}/month")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE! Model ready for production.")
    print("="*60)
    
    print("\n📍 Next steps:")
    print("   1. Start FastAPI server: uvicorn src.api:app --reload")
    print("   2. Launch dashboard: streamlit run src/dashboard.py")
    print("   3. Make predictions: curl -X POST http://127.0.0.1:8000/predict ...")


if __name__ == "__main__":
    main()

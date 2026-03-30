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
    """Extract and clean data directly from the profiler logic."""
    from scripts.data_profiler import extract_data
    df = extract_data()
    df = clean_listings_data(df)
    return df

def main():
    print("="*60)
    print("🚀 Model Training Pipeline initialized")
    print("="*60)
    
    print("\nLoading and cleaning data...")
    df = get_data()
    
    if df.empty:
        print("❌ Dataset is empty. Aborting training.")
        return

    # 1. Feature Separation
    # Note: 'price_per_room' is specifically excluded here to prevent TARGET LEAKAGE.
    # Since 'price_per_room' is derived directly from 'monthly_rent_eur' (the target), 
    # it cannot be used as an independent feature.
    
    features = ['neighborhood', 'rooms', 'is_pet_friendly']
    target = 'monthly_rent_eur'
    
    # Drop rows where target or critical features are missing implicitly by cleaning
    df = df.dropna(subset=[target])
    
    X = df[features]
    y = df[target]
    
    print(f"\n📊 Data shape: Features X={X.shape}, Target y={y.shape}")
    
    # 2. Data Splitting (80/20 train-test allocation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")
    
    # 3. Preprocessing & Sklearn Pipeline setup
    # FunctionTransformer to handle binary encoding via the registered function
    binary_transformer = FunctionTransformer(encode_binary, validate=False)
    
    # TargetEncoder for neighborhood high cardinality categorical encoding
    target_encoder = ce.TargetEncoder(cols=['neighborhood'])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('neighborhood_te', target_encoder, ['neighborhood']),
            ('pet_friendly_bin', binary_transformer, ['is_pet_friendly']),
            ('passthrough', 'passthrough', ['rooms'])
        ],
        remainder='drop'
    )
    
    # Group the XGBRegressor and preprocessing objects into a single pipeline object
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])
    
    # 4. Hyperparameter Tuning & Model Instantiation
    param_distributions = {
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6, 8],
        'model__n_estimators': [50, 100, 200, 300]
    }
    
    print("\n🔍 Starting RandomizedSearchCV for Hyperparameter Tuning...")
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=15,    # Number of parameter settings sampled
        cv=3,         # 3-fold cross validation
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_pipeline = random_search.best_estimator_
    print(f"✅ Best Parameters Found:\n   {random_search.best_params_}")
    
    # 5. Training & Evaluation
    print("\n📈 Evaluating Model on Unseen Test Set...")
    y_pred = best_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Mean Absolute Error (MAE):       €{mae:.2f}")
    print(f"   Root Mean Squared Error (RMSE):  €{rmse:.2f}")
    print(f"   R-squared (R²):                  {r2:.4f}")
    
    # 6. Serialization
    print("\n📦 Serializing trained pipeline to disk...")
    os.makedirs('assets', exist_ok=True)
    model_path = os.path.join('assets', 'model.joblib')
    joblib.dump(best_pipeline, model_path)
    print(f"✅ Model pipeline successfully serialized to {model_path}")
    print("="*60)

if __name__ == "__main__":
    main()

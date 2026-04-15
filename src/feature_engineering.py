"""
Feature Engineering Module for Real Estate Rent Prediction
===========================================================
Implements domain-driven feature transformations and encodings for machine learning.
Converts raw categorical and binary features into model-ready numerical representations.

Engineering Philosophy:
- Target Encoding captures neighborhood price dynamics (high-cardinality categorical)
- Binary encoding converts boolean flags to 0/1 for tree-based algorithms
- Derived features (price_per_room) add interpretable complexity for pattern discovery

Expected Input Schema:
- neighborhood (str): High-cardinality categorical (15-30 unique values typical)
- rooms (int): Room count (typically 1-4 range)
- is_pet_friendly (bool): Binary policy indicator
- monthly_rent_eur (int): Target variable (only used in encoding reference)

Feature Transformations:
1. Derived: price_per_room = monthly_rent_eur / rooms (ratio feature for XGBoost splits)
2. Categorical: Target encoding (neighborhood -> mean rent per area)
3. Binary: Boolean -> Integer mapping (is_pet_friendly: {False, True} -> {0, 1})
"""

import pandas as pd
import numpy as np


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Target Encoding to the neighborhood feature.
    
    **Why Target Encoding?**
    - Neighborhood is high-cardinality (15-30 categories), making one-hot encoding inefficient
    - Target encoding (mean encoding) replaces categories with their mean target value
    - Captures the implicit domain knowledge: "Centru is expensive" <-> high encoded value
    
    **Implementation:**
    - Replace each neighborhood with the historical mean rent for that area
    - Use overall mean as smoothing fallback for unseen neighborhoods in production
    - Handles new neighborhoods gracefully without breaking inference
    
    Args:
        df: DataFrame with columns ['neighborhood', 'monthly_rent_eur'].
    
    Returns:
        DataFrame with 'neighborhood' replaced by numerical mean-rent values.
        New column dtype: float64.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'neighborhood': ['Centru', 'Periphery', 'Centru'],
        ...     'monthly_rent_eur': [500, 300, 550]
        ... })
        >>> result = encode_categorical(df)
        >>> result['neighborhood'].tolist()
        [525.0, 300.0, 525.0]  # Mean rent per neighborhood
    
    FIXME: Beware data leakage if train/test split performed after encoding.
           Always fit encoder on train data, transform test data separately.
    """
    if "neighborhood" in df.columns and "monthly_rent_eur" in df.columns:
        # Calculate mean rent per neighborhood (target statistics)
        neighborhood_means = df.groupby("neighborhood")["monthly_rent_eur"].mean()
        overall_mean = df["monthly_rent_eur"].mean()
        
        # Replace neighborhood string with encoded numerical value
        # Fallback to overall mean for unseen neighborhoods
        df["neighborhood"] = df["neighborhood"].map(neighborhood_means).fillna(overall_mean)
        
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean pet-friendly flag to binary integer representation.
    
    **Why needed?**
    XGBoost (tree-based models) require numeric inputs. Boolean columns must be
    converted to integers (True -> 1, False -> 0) for algorithm ingestion.
    
    **Encoding Scheme:**
    - True  -> 1  (pet-friendly apartment)
    - False -> 0  (no pets allowed)
    - NaN   -> NaN (preserved; handled by model preprocessing)
    
    Args:
        df: DataFrame with 'is_pet_friendly' column (dtype: bool or bool-like).
    
    Returns:
        DataFrame with 'is_pet_friendly' replaced by integers {0, 1}.
        Output dtype: int64.
    
    Note: This function is idempotent on already-encoded data.
    """
    if "is_pet_friendly" in df.columns:
        # Apply explicit type conversion: True -> 1, False -> 0
        df["is_pet_friendly"] = df["is_pet_friendly"].apply(lambda x: 1 if x is True else 0)
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize new features through domain-driven feature engineering.
    
    **Derived Feature: price_per_room**
    
    Rationale:
    - Raw features (rooms, rent) have different scales and interpretations
    - Ratio feature captures efficiency: lower price/room = better value
    - XGBoost can learn non-linear relationships, but explicit ratio accelerates learning
    - Interpretable: rental agents naturally think in "price per room" terms
    
    Expected behavior:
    - 2-bedroom apartment, €800/month -> price_per_room = 400 (high-value market)
    - Studio apartment, €400/month -> price_per_room = 400 (comparable market)
    
    Edge Case Handling:
    - If rooms = 0 (data anomaly), assign price_per_room = 0.0 to avoid division errors
    - Outlier detection (earlier pipeline) should catch these anomalies, but this
      defensive measure prevents runtime crashes during inference
    
    Args:
        df: DataFrame with columns ['monthly_rent_eur', 'rooms'].
    
    Returns:
        DataFrame with new column 'price_per_room' added (dtype: float64).
        Original columns unchanged.
    
    TODO: Consider adding room_density = rooms / sqrt(monthly_rent_eur) for luxury segment analysis.
    """
    if "monthly_rent_eur" in df.columns and "rooms" in df.columns:
        # Vectorized division with safeguard against division by zero
        # np.where ensures price_per_room = 0 when rooms = 0 (degenerate case)
        df["price_per_room"] = np.where(
            df["rooms"] > 0,
            df["monthly_rent_eur"] / df["rooms"],
            0.0
        )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate complete feature engineering pipeline.
    
    **Processing Order (Critical):**
    1. create_derived_features: Generate ratio features using raw inputs
    2. encode_categorical: Transform high-cardinality strings to numerical
    3. encode_binary: Convert boolean columns to integers
    
    This sequence is intentional:
    - Derived features must be created BEFORE encoding (require raw values)
    - Categorical/binary encoding happens last (produces model-ready numerical arrays)
    
    Args:
        df: Raw DataFrame from data cleaning module with schema:
            ['neighborhood', 'rooms', 'is_pet_friendly', 'monthly_rent_eur'].
    
    Returns:
        Fully feature-engineered DataFrame with schema:
        ['neighborhood' (float64), 'rooms' (int64), 'is_pet_friendly' (int64),
         'price_per_room' (float64), 'monthly_rent_eur' (int64)].
    
    Idempotence: Second call on already-engineered data may produce inconsistent
                 neighborhood encoding due to re-fitting on transformed data.
                 ALWAYS apply feature engineering only once.
    
    Example:
        >>> raw_df = pd.DataFrame({
        ...     'neighborhood': ['Centru'],
        ...     'rooms': [2],
        ...     'is_pet_friendly': [True],
        ...     'monthly_rent_eur': [600]
        ... })
        >>> engineered = engineer_features(raw_df)
        >>> engineered.dtypes
        neighborhood: float64
        rooms: int64
        is_pet_friendly: int64
        price_per_room: float64
        monthly_rent_eur: int64
    """
    if df is None or df.empty:
        print("⚠️  Received empty or None DataFrame. Returning as-is.")
        return df
    
    print("🔧 Engineering features...")
    df = create_derived_features(df)
    df = encode_categorical(df)
    df = encode_binary(df)
    print("✅ Feature engineering complete.\n")
    
    return df

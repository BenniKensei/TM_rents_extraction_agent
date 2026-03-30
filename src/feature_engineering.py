import pandas as pd
import numpy as np

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target Encoding: Replace the neighborhood name with the historical
    average rent_eur for that specific area.
    """
    if "neighborhood" in df.columns and "monthly_rent_eur" in df.columns:
        # Calculate mean rent per neighborhood
        neighborhood_means = df.groupby("neighborhood")["monthly_rent_eur"].mean()
        overall_mean = df["monthly_rent_eur"].mean()
        
        # Replace the neighborhood name with the mathematical target encoded value
        df["neighborhood"] = df["neighborhood"].map(neighborhood_means).fillna(overall_mean)
        
    return df

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary Encoding: Map boolean flags (e.g., pet_friendly) directly to binary integers (1 for True, 0 for False).
    """
    if "is_pet_friendly" in df.columns:
        df["is_pet_friendly"] = df["is_pet_friendly"].apply(lambda x: 1 if x is True else 0)
    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived Features: Synthesize new predictive columns. 
    Calculate price_per_room.
    """
    if "monthly_rent_eur" in df.columns and "rooms" in df.columns:
        # Add a small epsilon to avoid division by zero if there's a 0 room anomaly
        # Or use np.where to assign 0 when rooms is 0
        df["price_per_room"] = np.where(df["rooms"] > 0, df["monthly_rent_eur"] / df["rooms"], 0.0)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to apply all feature engineering and encoding steps
    for downstream machine learning algorithm ingestion.
    """
    if df is None or df.empty:
        return df
        
    df = create_derived_features(df)
    df = encode_categorical(df)
    df = encode_binary(df)
    
    return df

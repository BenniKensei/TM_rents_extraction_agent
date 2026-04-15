"""
Data Cleaning Module for Real Estate Listings
==============================================
Implements robust data quality and preprocessing operations on web-scraped apartment listings.
Handles deduplication, missing value imputation, and statistical outlier detection.

This module assumes input DataFrames have the following schema:
- title (str): Apartment listing title
- monthly_rent_eur (int): Target variable - monthly rent in EUR
- neighborhood (str): Categorical location identifier
- rooms (int): Number of rooms (feature)
- is_pet_friendly (bool): Binary pet policy indicator
- last_seen (datetime, optional): Timestamp of last observation

Business Logic:
- Deduplication prioritizes temporal recency to handle listing updates
- Missing target values are dropped to maintain prediction integrity
- Outliers are detected via IQR to preserve market realism without aggressive truncation
"""

import pandas as pd


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eliminate duplicate listings by retaining only the most recent observation.
    
    Real estate listings are frequently re-scraped across multiple days. This function
    identifies duplicates based on (title, neighborhood) composite key and retains the
    most recent entry (latest last_seen timestamp) to capture price updates.
    
    Args:
        df: DataFrame with columns ['title', 'neighborhood', 'last_seen'].
    
    Returns:
        DataFrame with duplicates removed, keeping the temporally latest record per listing.
    
    Time Complexity: O(n log n) due to sort operation.
    Space Complexity: O(n) for the deduplicated result.
    """
    # Sort by last_seen ascending so that the last occurrence (most recent) is retained
    if "last_seen" in df.columns:
        df = df.sort_values(by="last_seen", ascending=True)
    
    # Drop duplicates based on composite key (title, neighborhood), keeping last (most recent)
    # Rationale: Multiple observations of the same listing indicate price updates or re-listings
    df = df.drop_duplicates(subset=["title", "neighborhood"], keep="last")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific missing value handling for apartment listings.
    
    **Critical Rule:** Drop rows where target variable (monthly_rent_eur) is missing.
    Prediction accuracy cannot be ensured without ground truth labels.
    
    For categorical features (neighborhood), impute missing values with 'Unknown' placeholder
    to preserve row count while flagging data quality issues downstream.
    
    Args:
        df: DataFrame with optional missing values in ['monthly_rent_eur', 'neighborhood'].
    
    Returns:
        DataFrame with missing target values removed and missing categorical values imputed.
    
    Raises:
        (No exceptions raised - operations are graceful on edge cases)
    """
    # Drop rows missing the target variable (no labels = unusable for training)
    if "monthly_rent_eur" in df.columns:
        rows_before = len(df)
        df = df.dropna(subset=["monthly_rent_eur"])
        rows_dropped = rows_before - len(df)
        if rows_dropped > 0:
            print(f"⚠️  Dropped {rows_dropped} rows missing target variable (monthly_rent_eur)")
    
    # Impute missing categorical features with placeholder to preserve cardinality information
    if "neighborhood" in df.columns:
        missing_count = df["neighborhood"].isna().sum()
        df["neighborhood"] = df["neighborhood"].fillna("Unknown")
        if missing_count > 0:
            print(f"⚠️  Imputed {missing_count} missing neighborhood values with 'Unknown'")
        
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and remove statistical outliers using the Interquartile Range (IQR) method.
    
    The IQR method identifies anomalies as values beyond [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    This approach is robust for skewed distributions and resistant to extreme values.
    
    Business Rationale:
    - monthly_rent_eur outliers: Studio apartments listed at €50/month or €10,000/month
      likely represent data entry errors or commercial properties (not apartments).
    - rooms outliers: Listings with 0 rooms or >8 rooms violate apartment market assumptions.
    
    Args:
        df: DataFrame with numeric columns ['monthly_rent_eur', 'rooms'].
    
    Returns:
        DataFrame with outliers removed. Rows with missing values in outlier columns
        are preserved (missing is not the same as anomalous).
    
    TODO: Consider robust scaling (Tukey Outliers + domain knowledge) for rent data
          with heavy right tail skew in luxury segment.
    """
    for col in ["monthly_rent_eur", "rooms"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Calculate IQR boundaries (robust to extreme values)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Keep rows that satisfy: value is NaN OR value is within [lower_bound, upper_bound]
            rows_before = len(df)
            df = df[(df[col].isna()) | ((df[col] >= lower_bound) & (df[col] <= upper_bound))]
            rows_after = len(df)
            
            if rows_before != rows_after:
                print(f"🔍 {col}: Removed {rows_before - rows_after} outliers "
                      f"(bounds: [{lower_bound:.0f}, {upper_bound:.0f}])")
            
    return df


def clean_listings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master data cleaning pipeline combining deduplication, missing value handling, and outlier removal.
    
    This function orchestrates the complete data quality workflow:
    1. Deduplicate: Remove temporal duplicates, keeping most recent observation
    2. Handle missing values: Drop incomplete target rows, impute missing features
    3. Remove outliers: Eliminate statistical anomalies via IQR
    
    Args:
        df: Raw input DataFrame from web scraper or database.
    
    Returns:
        Clean, production-ready DataFrame ready for feature engineering and modeling.
    
    Idempotence: Applying clean_listings_data twice yields identical output.
    
    Example:
        >>> raw_df = pd.read_csv('listings.csv')
        >>> clean_df = clean_listings_data(raw_df)
        >>> assert len(clean_df) <= len(raw_df)  # Monotonically decreasing row count
    """
    if df is None or df.empty:
        print("⚠️  Received empty or None DataFrame. Returning as-is.")
        return df
    
    print(f"📊 Starting data cleaning pipeline ({len(df)} input rows)...")
    df = deduplicate(df)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    print(f"✅ Cleaning complete. Output: {len(df)} rows\n")
    
    return df

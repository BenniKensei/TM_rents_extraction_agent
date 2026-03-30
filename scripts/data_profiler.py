"""
Data Extraction & Profiling Pipeline
=====================================
Extracts timisoara_rents records from Supabase PostgreSQL and performs
comprehensive profiling: schema analysis, null distributions, and anomalies.

Supports:
- Supabase PostgreSQL (primary)
- Local Docker PostgreSQL (fallback)
- Sample data generation (demonstration)
"""

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.data_cleaning import clean_listings_data
    from src.feature_engineering import engineer_features
except ImportError:
    def clean_listings_data(df): return df
    def engineer_features(df): return df

load_dotenv()


def connect_database(use_local: bool = False) -> Optional[psycopg2.extensions.connection]:
    """
    Establish connection to PostgreSQL database.
    
    Args:
        use_local: If True, tries to connect to local Docker PostgreSQL.
                   If False, tries Supabase first, then falls back to local.
    
    Returns:
        Database connection or None if connection fails.
    """
    if use_local:
        # Try local Docker PostgreSQL
        print("🔗 Connecting to local Docker PostgreSQL...")
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5433,
                user="Ben",
                password="postgres88",
                database="WebExAg"
            )
            print("✅ Connected to local PostgreSQL")
            return conn
        except Exception as e:
            print(f"⚠️  Local connection failed: {e}")
            return None
    else:
        # Try Supabase first
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        print("🔗 Connecting to Supabase PostgreSQL...")
        try:
            conn = psycopg2.connect(database_url)
            print("✅ Connected to Supabase")
            return conn
        except Exception as e:
            print(f"⚠️  Supabase connection failed: {e}")
            print("🔗 Attempting fallback to local Docker PostgreSQL...")
            return connect_database(use_local=True)




def extract_data(use_sample: bool = False) -> pd.DataFrame:
    """
    Extract historical timisoara_rents records into Pandas DataFrame.
    
    Args:
        use_sample: If True, generates sample data for demonstration.
                    If False, attempts to query actual database.
    
    Returns:
        DataFrame with extracted records.
    """
    if use_sample:
        print("\n📊 Generating sample timisoara_rents data for demonstration...")
        df = generate_sample_data()
        print(f"✅ Generated {len(df)} sample records")
        return df
    
    conn = connect_database()
    if conn is None:
        print("⚠️  No database connection available. Using sample data for demonstration...")
        return generate_sample_data()
    
    query = """
    SELECT 
        id,
        title,
        monthly_rent_eur,
        neighborhood,
        rooms,
        is_pet_friendly,
        first_seen,
        last_seen
    FROM timisoara_rents
    ORDER BY first_seen DESC;
    """
    
    print("\n📊 Extracting timisoara_rents records...")
    try:
        df = pd.read_sql(query, conn)
        print(f"✅ Extracted {len(df)} records")
        conn.close()
        return df
    except Exception as e:
        print(f"⚠️  Query failed: {e}. Using sample data for demonstration...")
        conn.close()
        return generate_sample_data()


def generate_sample_data() -> pd.DataFrame:
    """Generate realistic sample apartment listing data for profiling demonstration."""
    np.random.seed(42)
    random.seed(42)
    
    neighborhoods = [
        "Complexul Studentesc", "Centru", "Iosefin", "Buziaș", 
        "Mehala", "Micălaca", "Plopi", "Girocului", "Dambovita",
        "Fabric", "Tipografiei", "Moghioroș", "Dorobanți"
    ]
    
    titles = [
        f"Apartament {random.randint(1, 4)} camere - {neighborhood}"
        for neighborhood in neighborhoods
        for _ in range(random.randint(8, 15))
    ]
    titles = list(set(titles))  # Remove duplicates
    
    n_records = random.randint(150, 250)
    
    data = {
        "id": list(range(1, n_records + 1)),
        "title": random.choices(titles, k=n_records),
        "monthly_rent_eur": [
            int(np.random.normal(loc=300, scale=80)) 
            for _ in range(n_records)
        ],
        "neighborhood": random.choices(neighborhoods, k=n_records),
        "rooms": np.random.choice([1, 2, 3, 4], size=n_records, p=[0.2, 0.4, 0.3, 0.1]),
        "is_pet_friendly": random.choices(
            [True, False, None], 
            k=n_records, 
            weights=[0.15, 0.75, 0.1]
        ),
        "first_seen": [
            datetime.now() - timedelta(days=random.randint(0, 90))
            for _ in range(n_records)
        ],
        "last_seen": [
            datetime.now() - timedelta(days=random.randint(0, 30))
            for _ in range(n_records)
        ]
    }
    
    df = pd.DataFrame(data)
    # Introduce some realistic anomalies
    df.loc[random.sample(range(len(df)), min(5, len(df))), "monthly_rent_eur"] = \
        df.loc[random.sample(range(len(df)), min(5, len(df))), "monthly_rent_eur"] * 3
    
    return df


def analyze_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze and map schema to identify feature data types."""
    schema_info = {
        "column_count": len(df.columns),
        "row_count": len(df),
        "columns": {}
    }
    
    print("\n📋 SCHEMA ANALYSIS")
    print("=" * 70)
    print(f"Total Columns: {schema_info['column_count']}")
    print(f"Total Rows: {schema_info['row_count']}")
    print("\nColumn Details:")
    print("-" * 70)
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_percent = (null_count / len(df) * 100) if len(df) > 0 else 0
        
        schema_info["columns"][col] = {
            "dtype": dtype,
            "non_null": int(non_null_count),
            "null": int(null_count),
            "null_percent": round(null_percent, 2),
            "memory_usage": df[col].memory_usage(deep=True)
        }
        
        print(f"  {col:20} | Type: {dtype:12} | Non-null: {non_null_count:6} | Null: {null_count:6} ({null_percent:5.2f}%)")
    
    return schema_info


def analyze_null_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze distribution of null values across features."""
    null_analysis = {}
    
    print("\n❌ NULL VALUE DISTRIBUTION")
    print("=" * 70)
    
    feature_cols = ["monthly_rent_eur", "neighborhood", "rooms", "is_pet_friendly"]
    
    for col in feature_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_percent = (null_count / len(df) * 100) if len(df) > 0 else 0
            
            null_analysis[col] = {
                "null_count": int(null_count),
                "null_percent": round(null_percent, 2),
                "non_null_count": int(df[col].notna().sum())
            }
            
            status = "⚠️ " if null_percent > 5 else "✅"
            print(f"{status} {col:20} | Null: {null_count:6} ({null_percent:6.2f}%)")
    
    return null_analysis


def analyze_numeric_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical analysis on numeric features."""
    numeric_analysis = {}
    
    print("\n📊 NUMERIC FEATURES ANALYSIS")
    print("=" * 70)
    
    numeric_cols = ["rooms", "monthly_rent_eur"]
    
    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            data = df[col].dropna()
            
            if len(data) > 0:
                stats = {
                    "count": len(data),
                    "mean": float(data.mean()),
                    "median": float(data.median()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "q1": float(data.quantile(0.25)),
                    "q3": float(data.quantile(0.75)),
                    "iqr": float(data.quantile(0.75) - data.quantile(0.25)),
                }
                
                numeric_analysis[col] = stats
                
                print(f"\n  {col.upper()}")
                print(f"    Count:  {stats['count']}")
                print(f"    Mean:   {stats['mean']:.2f}")
                print(f"    Median: {stats['median']:.2f}")
                print(f"    Std:    {stats['std']:.2f}")
                print(f"    Min:    {stats['min']:.2f}")
                print(f"    Q1:     {stats['q1']:.2f}")
                print(f"    Q3:     {stats['q3']:.2f}")
                print(f"    IQR:    {stats['iqr']:.2f}")
                print(f"    Max:    {stats['max']:.2f}")
    
    return numeric_analysis


def detect_anomalies(df: pd.DataFrame, numeric_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Detect statistical anomalies using IQR method."""
    anomalies = {}
    
    print("\n🚨 ANOMALY DETECTION (IQR Method)")
    print("=" * 70)
    
    for col in numeric_stats.keys():
        if col in df.columns:
            stats = numeric_stats[col]
            q1 = stats["q1"]
            q3 = stats["q3"]
            iqr = stats["iqr"]
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(df) * 100) if len(df) > 0 else 0
            
            anomalies[col] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": int(outlier_count),
                "outlier_percent": round(outlier_percent, 2),
                "outlier_values": outliers[col].tolist() if outlier_count > 0 else []
            }
            
            print(f"\n  {col.upper()}")
            print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"    Outliers: {outlier_count} ({outlier_percent:.2f}%)")
            
            if outlier_count > 0:
                print(f"    Outlier values (sample): {outliers[col].head(5).tolist()}")
    
    return anomalies


def analyze_categorical_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze categorical features (neighborhood, pet_friendly)."""
    categorical_analysis = {}
    
    print("\n🏷️  CATEGORICAL FEATURES ANALYSIS")
    print("=" * 70)
    
    # Neighborhood analysis
    if "neighborhood" in df.columns:
        neighborhood_counts = df["neighborhood"].value_counts()
        
        categorical_analysis["neighborhood"] = {
            "unique_values": int(df["neighborhood"].nunique()),
            "top_5": neighborhood_counts.head(5).to_dict(),
            "distribution": neighborhood_counts.to_dict()
        }
        
        print("\n  NEIGHBORHOOD")
        print(f"    Unique Values: {categorical_analysis['neighborhood']['unique_values']}")
        print("    Top 5 Neighborhoods:")
        for neighborhood, count in neighborhood_counts.head(5).items():
            percent = (count / len(df) * 100)
            print(f"      • {neighborhood}: {count} ({percent:.2f}%)")
    
    # Pet-friendly analysis
    if "is_pet_friendly" in df.columns:
        pet_counts = df["is_pet_friendly"].value_counts(dropna=False)
        
        categorical_analysis["is_pet_friendly"] = {
            "true_count": int(pet_counts.get(True, 0)),
            "false_count": int(pet_counts.get(False, 0)),
            "null_count": int(pet_counts.get(np.nan, 0)),
            "distribution": pet_counts.to_dict()
        }
        
        print("\n  PET-FRIENDLY")
        print(f"    True:  {pet_counts.get(True, 0)}")
        print(f"    False: {pet_counts.get(False, 0)}")
        print(f"    Null:  {pet_counts.get(np.nan, 0)}")
    
    return categorical_analysis


def generate_profile_report(
    schema_info: Dict,
    null_analysis: Dict,
    numeric_stats: Dict,
    anomalies: Dict,
    categorical_analysis: Dict,
    df: pd.DataFrame
) -> None:
    """Generate final profiling report."""
    print("\n" + "=" * 70)
    print("📈 PROFILING SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\n✅ Dataset Size: {len(df)} records across {len(df.columns)} columns")
    print(f"⏱️  Date Range: {df['first_seen'].min()} to {df['first_seen'].max()}")
    
    # Data quality score
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.notna().sum().sum()
    data_quality_score = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
    
    print(f"📊 Data Quality Score: {data_quality_score:.2f}%")
    
    # Memory usage
    total_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"💾 Memory Usage: {total_memory:.2f} MB")
    
    print("\n" + "=" * 70)


def main():
    """Main execution pipeline."""
    try:
        print("\n" + "=" * 70)
        print("🚀 DATA EXTRACTION & PROFILING PIPELINE")
        print("=" * 70)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if --use-sample flag is provided
        use_sample = "--use-sample" in sys.argv or "--demo" in sys.argv
        
        # Extract data
        df = extract_data(use_sample=use_sample)
        df = clean_listings_data(df)
        
        if len(df) == 0:
            print("\n⚠️  No records found in timisoara_rents table")
            return
        
        # Perform analyses
        schema_info = analyze_schema(df)
        null_analysis = analyze_null_distribution(df)
        numeric_stats = analyze_numeric_features(df)
        anomalies = detect_anomalies(df, numeric_stats)
        categorical_analysis = analyze_categorical_features(df)
        
        # Generate report
        generate_profile_report(
            schema_info,
            null_analysis,
            numeric_stats,
            anomalies,
            categorical_analysis,
            df
        )
        
        print(f"\n✅ Profiling completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Apply feature engineering right before CSV export to provide ML-ready data
        print("\n⚙️  Applying Feature Engineering for Model Export...")
        ml_df = engineer_features(df.copy())
        
        # Save to CSV for reference
        output_file = "timisoara_rents_extracted.csv"
        ml_df.to_csv(output_file, index=False)
        print(f"💾 ML-ready data exported to: {output_file}")
        
        # Create detailed profiling report JSON
        import json
        report = {
            "extraction_time": datetime.now().isoformat(),
            "record_count": len(df),
            "column_count": len(df.columns),
            "schema": {
                col: {
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isna().sum()),
                    "null_percent": float((df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0)
                }
                for col in df.columns
            },
            "null_distribution": null_analysis,
            "numeric_statistics": {k: v for k, v in numeric_stats.items()},
            "anomalies": {k: v for k, v in anomalies.items()},
            "categorical_analysis": categorical_analysis
        }
        
        report_file = "timisoara_profiling_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📋 Detailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

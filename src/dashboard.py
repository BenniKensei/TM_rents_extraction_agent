import streamlit as st
import pandas as pd
import psycopg2
import os
import requests
import joblib
from dotenv import load_dotenv
from data_cleaning import clean_listings_data
from pathlib import Path

load_dotenv()

st.set_page_config(
    page_title="Timișoara Real Estate Market Analytics", page_icon="🏠", layout="wide"
)

st.title("Timișoara Real Estate Market Analytics")


@st.cache_resource
def get_db_connection():
    """Establish and cache a connection to the PostgreSQL database URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        st.error("Error: DATABASE_URL not found in environment variables.")
        return None

    try:
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None


@st.cache_data(ttl=60)
def load_data():
    """Fetch all records from the timisoara_rents table or local CSV fallback."""
    conn = get_db_connection()
    
    # Define fallback mechanism
    def load_fallback():
        csv_path = "timisoara_rents_extracted.csv"
        if os.path.exists(csv_path):
            st.info("⚠️ Falling back to local CSV file due to database connection issues.")
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    if conn is None:
        return load_fallback()

    query = "SELECT * FROM timisoara_rents;"
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return load_fallback()


@st.cache_resource
def load_model_neighborhoods():
    """Extract the neighborhood vocabulary from the trained inference model."""
    model_path = Path(__file__).resolve().parents[1] / "assets" / "model.joblib"
    if not model_path.exists():
        return []

    try:
        model = joblib.load(model_path)
        preprocessor = model.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["neighborhood_te"]

        neighborhoods = []
        for mapping_info in getattr(encoder.ordinal_encoder, "category_mapping", []):
            mapping = mapping_info.get("mapping")
            if hasattr(mapping, "index"):
                neighborhoods.extend(
                    [
                        str(value)
                        for value in list(mapping.index)
                        if str(value).lower() != "nan"
                    ]
                )

        return sorted(dict.fromkeys(neighborhoods))
    except Exception:
        return []


# Load the listings data
df = load_data()
df = clean_listings_data(df)

if df.empty:
    st.warning("No data found in the `timisoara_rents` table.")
else:
    # High-level metrics
    total_listings = len(df)
    avg_rent = df["monthly_rent_eur"].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Active Listings", f"{total_listings:,}")
    with col2:
        # Format the average rent as EUR
        st.metric("Average Rent (EUR)", f"€{avg_rent:,.2f}")

    st.markdown("---")

    # Bar chart: Average Rent by Neighborhood
    st.subheader("Average Rent by Neighborhood")

    # Calculate mean rent grouped by neighborhood
    avg_rent_by_nbh = (
        df.groupby("neighborhood")["monthly_rent_eur"].mean().reset_index()
    )
    avg_rent_by_nbh = avg_rent_by_nbh.sort_values(
        by="monthly_rent_eur", ascending=False
    )

    # Use Streamlit's built-in bar chart visualization tool
    st.bar_chart(
        avg_rent_by_nbh,
        x="neighborhood",
        y="monthly_rent_eur",
        use_container_width=True,
    )

    st.markdown("---")

    # Interactive Data Table
    st.subheader("Raw Listings Data")

    # Sort the dataframe by last_seen descending if it exists
    if "last_seen" in df.columns:
        df_sorted = df.sort_values(by="last_seen", ascending=False)
    else:
        df_sorted = df

    st.dataframe(df_sorted, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Interactive ML Prediction Block
    st.subheader("🤖 AI Rent Price Estimator")
    st.write("Enter apartment specifications to evaluate if the asking price is a Great Deal, Fair, or Overpriced based on our XGBoost model.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            # Prefer the model's trained neighborhood vocabulary so predictions vary meaningfully.
            model_neighborhoods = load_model_neighborhoods()
            neighborhoods = df['neighborhood'].dropna().astype(str).unique().tolist()
            neighborhoods = sorted(set(neighborhoods).union(model_neighborhoods))

            if not neighborhoods:
                neighborhoods = ["Centru", "Complexul Studentesc", "Iosefin", "Mehala"]

            if model_neighborhoods:
                st.caption(
                    "Neighborhood options are taken from the trained model so the prediction changes across areas it actually knows."
                )
            
            default_index = neighborhoods.index("Centru") if "Centru" in neighborhoods else 0
            sc_neighborhood = st.selectbox("Neighborhood", neighborhoods, index=default_index)
            sc_rooms = st.slider("Rooms", 1, 6, 2)
            
        with col_b:
            sc_asking_price = st.number_input("Asking Price (EUR)", min_value=50, max_value=5000, value=400, step=10)
            sc_pet = st.checkbox("Is Pet Friendly?", value=False)
            
        submit_btn = st.form_submit_button("Evaluate Price")

    if submit_btn:
        api_url = "http://127.0.0.1:8000/predict"
        payload = {
            "neighborhood": sc_neighborhood,
            "rooms": sc_rooms,
            "is_pet_friendly": sc_pet
        }
        
        try:
            with st.spinner("Executing mathematical assessment..."):
                response = requests.post(api_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                pred_price = response.json().get("predicted_rent_eur", 0.0)
                
                st.markdown(f"#### **Predicted Fair Market Rent:** €{pred_price:,.2f}")
                
                # Formulate structural boundaries natively (+/- 10%)
                max_fair = pred_price * 1.10
                min_fair = pred_price * 0.90
                
                delta = sc_asking_price - pred_price
                delta_pct = (delta / pred_price) * 100 if pred_price > 0 else 0
                
                if sc_asking_price > max_fair:
                    st.error(f"🚨 **Overpriced** by ~{delta_pct:.1f}% (Asking: €{sc_asking_price} vs Predicted: €{pred_price:,.2f})")
                elif sc_asking_price < min_fair:
                    st.success(f"🔥 **Great Deal!** Below market by ~{abs(delta_pct):.1f}% (Asking: €{sc_asking_price} vs Predicted: €{pred_price:,.2f})")
                else:
                    st.info(f"⚖️ **Fair Price**. Bound within the standard ±10% local market range (Asking: €{sc_asking_price} vs Predicted: €{pred_price:,.2f})")
            else:
                st.error(f"Error mapped from backend inference API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot resolve the FASTAPI Prediction service. Please ensure `uvicorn src.api:app --reload` is running bound to port 8000.")

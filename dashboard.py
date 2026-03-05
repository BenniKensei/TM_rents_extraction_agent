import streamlit as st
import pandas as pd
import psycopg2
import os

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
    """Fetch all records from the timisoara_rents table."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()

    query = "SELECT * FROM timisoara_rents;"
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


# Load the listings data
df = load_data()

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

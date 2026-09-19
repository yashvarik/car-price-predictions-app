from pathlib import Path

import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# stream.py -> frontend/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "car_price.pkl"
DATA_PATH = BASE_DIR / "clean_car.csv"  # already-cleaned data, not the raw quikr_car.csv


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(subset=["name", "company", "fuel_type"])
    df["year"] = df["year"].astype(int)
    return df


pipe = load_model()
df = load_data()

st.title("🚗 Car Price Predictor")
st.caption("Used car ki estimated resale price - Quikr data pe trained")

companies = sorted(df["company"].unique())
years = sorted(df["year"].unique(), reverse=True)
fuel_types = sorted(df["fuel_type"].unique())

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", companies)
    available_models = sorted(df[df["company"] == company]["name"].unique())
    name = st.selectbox("Model", available_models)
    fuel_type = st.selectbox("Fuel type", fuel_types)

with col2:
    year = st.selectbox("Purchase year", years)
    kms_driven = st.number_input(
        "Kilometers driven", min_value=0, max_value=500000, value=40000, step=1000
    )

if st.button("Predict price", type="primary", use_container_width=True):
    query = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
    )
    price = float(pipe.predict(query)[0])

    if price <= 0:
        st.warning(
            f"Model ne ₹{price:,.0f} predict kiya - linear regression is combination pe "
            "negative ja raha hai. Rare/unseen model ho sakta hai."
        )
    else:
        st.success(f"Estimated price: ₹{price:,.0f}")
        st.caption("Ye ek estimate hai, exact market value nahi.")

with st.expander("Dataset ke baare me"):
    st.write(f"Rows: {len(df)} | Companies: {len(companies)} | Models: {len(df['name'].unique())}")
    st.dataframe(df.head(20), use_container_width=True)
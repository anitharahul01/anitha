import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    r"C:\Users\rahul\OneDrive\Desktop\anitha\DSA\dsa assignments\streamlitappdeploy\beer-servings.csv"
)
df.dropna(inplace=True)

# Load model
with open(
    r"C:\Users\rahul\OneDrive\Desktop\anitha\DSA\dsa assignments\streamlitappdeploy\model.pkl",
    "rb",
) as f:
    model = pickle.load(f)


st.title("Alcohol Consumption Predictor")
st.markdown(
    "Predict **total_litres_of_pure_alcohol** using beverage servings and region."
)


st.subheader(" Data Infographics")

# 1. Bar chart by continent
avg_alcohol = df.groupby("continent")["total_litres_of_pure_alcohol"].mean()
st.bar_chart(avg_alcohol)

# 2. Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- User Input Form ---
st.subheader("🧾 Input Features")

country = st.selectbox("Select Country", sorted(df["country"].unique()))
continent = st.selectbox("Select Continent", sorted(df["continent"].unique()))
beer_servings = st.number_input("Beer Servings", 0, 400, 100)
spirit_servings = st.number_input("Spirit Servings", 0, 400, 50)
wine_servings = st.number_input("Wine Servings", 0, 400, 30)

# --- Predict ---
if st.button("Predict"):
    input_df = pd.DataFrame(
        {
            "country": [country],
            "beer_servings": [beer_servings],
            "spirit_servings": [spirit_servings],
            "wine_servings": [wine_servings],
            "continent": [continent],
        }
    )

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Alcohol Consumption: {prediction:.2f} litres")

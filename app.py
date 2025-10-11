# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🌟 My First Streamlit App")
st.write("Hello! This is my first data app in Streamlit 🎈")

# Upload a CSV file
uploaded_file = st.file_uploader(
    r"C:\Users\rahul\OneDrive\Desktop\anitha\DSA\analytics vidya\food demand\train\train.csv",
    type=["csv"],
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    # Basic statistics
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Choose a column to plot
    numeric_columns = df.select_dtypes(include=["float", "int"]).columns
    if len(numeric_columns) > 0:
        col = st.selectbox("Pick a numeric column to plot:", numeric_columns)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20)
        st.pyplot(fig)
    else:
        st.write("No numeric columns found.")

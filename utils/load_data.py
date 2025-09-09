  
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path="H:/Home Dashboard/preprocessing/reduced_dataset.csv"):
    df = pd.read_csv(file_path)
    return df
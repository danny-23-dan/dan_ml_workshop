import streamtit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title(" PragyanAI Taxi Fare Prediction App (End-to-End ML)")
@st.cache data
def load data():
  url = "taxis.csv"
  df = pd.read_csv(url)
  df = df.convert_dtypes()
  st.write(df.head())
  return df
df = load_data()
st.subheader("PragyanAI Dataset Preview")

df = df[['distance', 'fare']].dropna()
df['distance"]= pd.to_numeric(df['distance'], errors='coerce')
df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
x = df[['distance']]
y = df['fare']

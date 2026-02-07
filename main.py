import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.write("""
# Housing Price Prediction
""")
rng = np.random.default_rng()

with open("finalized_model.pkl", "rb") as f:
    model = pickle.load(f)




# Create two columns for a more compact layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏠 Property Details")
    rm = st.slider("Average Number of Rooms", 3.0, 9.0, 6.0, step=0.1)
    age = st.slider("Proportion of Older Homes (Built <1940)", 0.0, 100.0, 50.0)
    dis = st.slider("Distance to Employment Centers (miles)", 1.0, 13.0, 4.0, step=0.1)
    chas = False
    
    st.markdown("### 🏫 Neighborhood & Education")
    ptratio = 11
    lstat = st.slider("Lower Socioeconomic Status (%)", 1.0, 40.0, 12.0, 
                      help="Percent of population with lower status")
    b = 200

with col2:
    st.markdown("### 🏙️ Location & Environment")
    crim = st.slider("Local Crime Rate", 0.0, 90.0, 3.6)
    zn = 50
    indus = 15
    nox = st.slider("Air Quality (Nitric Oxides Concentration)", 0.3, 0.9, 0.5, step=0.01)
    
    st.markdown("### 💰 Access & Taxes")
    rad = 12
    tax = st.slider("Property Tax Rate (per $10k)", 180, 720, 400)

pred = str((model.predict(np.array([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]).reshape(1, -1))*1000).astype(int)).strip("[]")
st.write("""## House Price:""")
st.write(f"""# ${pred}""")

st.write("""
#### By Atharva Gupta
""")

st.info("""
This model is trained on data from boston in the 1940s. It's predictions do not apply today. Use with caution
""")
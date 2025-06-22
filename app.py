# streamlit.py

import streamlit as st
import pandas as pd
from linear_regress import model

# App title
st.title("🏏 Shubman Gill - Next Innings Score Predictor")

# User inputs
opp_team = st.number_input("Opposition Team (1-5)", min_value=1, max_value=5, step=1)
match_location = st.radio("Match Location", ("Home", "Away"))
recent_form = st.slider("Recent Form (Average of last 5 innings)", 0.0, 100.0, 50.0)
previous_score = st.slider("Previous Match Score", 0.0, 150.0, 50.0)

# Prepare input DataFrame
input_df = pd.DataFrame({
    'Opposition_Team': [opp_team],
    'Match_Location': [1 if match_location == "Home" else 0],
    'Recent_Form': [recent_form],
    'Previous_Score': [previous_score]
})

# Predict button
if st.button("Predict Score"):
    predicted_score = model.predict(input_df)
    st.success(f"Predicted Score: {predicted_score[0]:.2f} runs")

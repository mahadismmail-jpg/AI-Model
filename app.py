import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page title and layout
st.set_page_config(page_title="McDonald's Calorie Predictor", layout="centered")

# App title and description
st.title("🍔 McDonald's Item Calorie Predictor")
st.write("Predict the calorie/energy content of McDonald's menu items using Machine Learning (Decision Tree Model).")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('mcd_decision_tree_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'mcd_decision_tree_model.pkl' not found. Please upload it to the app directory.")
        return None

dt_model = load_model()

if dt_model is not None:
    st.header("Input Nutritional Information:")
    
    # Input features corresponding to: ['protein', 'total_fat', 'sat_fat', 'trans_fat', 'chol', 'carbs', 'total_sugar', 'added_sugar', 'sodium']
    protein = st.slider("Protein (g)", 0.0, 50.0, 15.0, 0.1)
    total_fat = st.slider("Total Fat (g)", 0.0, 50.0, 14.0, 0.1)
    sat_fat = st.slider("Saturated Fat (g)", 0.0, 25.0, 5.0, 0.1)
    trans_fat = st.slider("Trans Fat (g)", 0.0, 5.0, 0.1, 0.01)
    chol = st.slider("Cholesterol (mg)", 0.0, 150.0, 30.0, 0.1)
    carbs = st.slider("Carbohydrates (g)", 0.0, 100.0, 35.0, 0.1)
    total_sugar = st.slider("Total Sugar (g)", 0.0, 100.0, 10.0, 0.1)
    added_sugar = st.slider("Added Sugar (g)", 0.0, 50.0, 5.0, 0.1)
    sodium = st.slider("Sodium (mg)", 0.0, 2000.0, 400.0, 10.0)

    # Predict button
    if st.button("Predict Calories"):
        # Prepare input features as a dataframe matching the training features
        input_features = pd.DataFrame([[
            protein, total_fat, sat_fat, trans_fat, chol, carbs, total_sugar, added_sugar, sodium
        ]], columns=['protein', 'total_fat', 'sat_fat', 'trans_fat', 'chol', 'carbs', 'total_sugar', 'added_sugar', 'sodium'])
        
        # Make prediction
        prediction = dt_model.predict(input_features)[0]
        
        # Determine class
        if prediction < 200:
            category = "Low"
            color = "green"
        elif prediction <= 500:
            category = "Medium"
            color = "orange"
        else:
            category = "High"
            color = "red"
            
        # Output prediction
        st.success(f"Predicted Energy: **{prediction:.2f} kcal**")
        st.markdown(f"Calorie Class: <span style='color:{color}; font-weight:bold;'>{category}</span>", unsafe_allow_html=True)

import streamlit as st
import joblib
import pandas as pd
import sklearn

model = joblib.load("model_preference_rf.pkl")

#Homepage
def home():
    st.title("Beach vs Mountain Travel Preference Predictor")
    st.markdown("""Welcome! Discover whether you are more likely to prefer Beach or Mountain destinations based on your personal and lifestyle information.""")
    st.subheader("📊 Features Used in Prediction")
    st.markdown("""
        - **Demographics**: Age, Gender, Education Level, Income, Location  
        - **Lifestyle**: Travel Frequency, Vacation Budget, Pets, Environmental Concerns
        - **Preferences**: Favorite Season, Preferred Activities  
        - **Environment**: Proximity to Mountains, Proximity to Beaches 
    """)
    
    st.subheader("How to Use")
    st.markdown("""
        1. Go to **Machine Learning** page  
        2. Fill in your personal information  
        3. Click **Predict**  
        4. See your predicted travel preference
    """)
    st.subheader("⚠️ Disclaimer")

    st.info("""
      This prediction is based on patterns learned from historical survey data.
      It should be used for educational and exploratory purposes only.
    """)

#User Inputs    
def machine_learning():
    st.title("Machine Learning Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Model Type")
        st.write("Random Forest Classifier")
    with col2:
        st.markdown("### Target")
        st.write("Beach vs Mountain Preference")
    with col3:
        st.markdown("### Features Used")
        st.write("13")
        
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        gender = st.selectbox("Gender", ["male", "female", "non-binary"])
        education = st.selectbox(
            "Education Level",
            ["high school", "bachelor", "master", "doctorate"]
        )
        location = st.selectbox(
            "Location",
            ["urban", "suburban", "rural"]
        )
        income = st.number_input("Income (Annual)", min_value=0, value=50000)
        travel_freq = st.number_input(
            "Travel Frequency (vacations per year)",
            min_value=0,
            max_value=20,
            value=2
        )
        vacation_budget = st.number_input("Vacation Budget", min_value=0, value=3000)
        favorite_season = st.selectbox(
            "Favorite Season",
            ["summer", "winter", "spring", "fall"]
        )
        preferred_activities = st.selectbox(
            "Preferred Activities",
            ["hiking", "swimming", "skiing", "sunbathing"]
        )
        proximity_mountains = st.number_input(
            "Proximity to Mountains (miles)",
            min_value=0.0,
            value=50.0,
            step=1.0
        )
        proximity_beaches = st.number_input(
            "Proximity to Beaches (miles)",
            min_value=0.0,
            value=50.0,
            step=1.0
        )
        pets = st.radio("Do you own pets?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        environmental_concerns = st.radio("Environmental Concerns", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Income": [income],
            "Travel_Frequency": [travel_freq],
            "Vacation_Budget": [vacation_budget],
            "Proximity_to_Mountains": [proximity_mountains],
            "Proximity_to_Beaches": [proximity_beaches],
            "Pets": [pets],
            "Environmental_Concerns": [environmental_concerns],
            "Location": [location],
            "Favorite_Season": [favorite_season],
            "Preferred_Activities": [preferred_activities],
            "Education_Level": [education]
        })

        prediction = model.predict(input_df)
        #st.write(prediction, type(prediction))
        #st.write("Type:", type(prediction))
        #st.write("Shape:", getattr(prediction, "shape", "no shape"))
        result = prediction.item()
        
        #this one is optional
        label_map = {0: "Mountain", 1: "Beach"}
        result = label_map.get(result, result)
        st.success(f"Prediction Result: {result}")

#Menu Settings        
def main():
    tab1, tab2 = st.tabs(["Home", "Machine Learning"])
    with tab1:
        home()
    with tab2:
        machine_learning()

if __name__ == "__main__":
    main()
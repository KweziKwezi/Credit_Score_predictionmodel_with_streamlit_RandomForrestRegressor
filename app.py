import streamlit as st
import joblib
import pandas as pd


st.set_page_config(page_title="Credit Score Predictor", page_icon="📊", layout="centered")


st.title("Credit Score Prediction Model")
st.markdown("**Random Forest Regressor** | Accuracy: **6.05% MAPE** | Avg Error: **~38 points**")
st.markdown("---")


@st.cache_resource
def load_model():
    return joblib.load('Final_model_for_credit_score_prediction')

model = load_model()


st.subheader("👤 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=28)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master"])
    person_income = st.number_input("Annual Income ($)", min_value=0, value=65000, step=1000)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, value=4)

with col2:
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0.0, value=5.0, step=0.5)
    previous_loan_defaults = st.selectbox("Previous Loan Defaults", ["No", "Yes"])


if st.button("🎯 Predict Credit Score", type="primary", use_container_width=True):
   
    new_applicant = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_home_ownership': [person_home_ownership],
        'previous_loan_defaults_on_file': [previous_loan_defaults]
    })
    
    
    encoded_applicant = pd.get_dummies(new_applicant).reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )
    
   
    predicted_score = model.predict(encoded_applicant)[0]
    
  
    st.markdown("---")
    st.subheader("📈 Prediction Result")
    
   
    if predicted_score >= 750:
        tier = "Excellent 🌟"
        color = "green"
    elif predicted_score >= 700:
        tier = "Good ✅"
        color = "blue"
    elif predicted_score >= 650:
        tier = "Fair ⚠️"
        color = "orange"
    elif predicted_score >= 600:
        tier = "Poor ⚠️"
        color = "orange"
    else:
        tier = "Very Poor ❌"
        color = "red"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Credit Score", f"{predicted_score:.0f}")
    with col2:
        st.metric("Credit Tier", tier)
    with col3:
        st.metric("Expected Range", f"{predicted_score-38:.0f} - {predicted_score+38:.0f}")
    
    st.info(f"💡 **Note:** This prediction has an average margin of error of ±38 points (6.05%).")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "<p>Built with by Kwezi | <a href='https://github.com/KweziKwezi' target='_blank'>GitHub</a></p>"
    "</div>",
    unsafe_allow_html=True
)
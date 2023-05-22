import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']

def show_predict_page():  # sourcery skip
    st.title("Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden"
    )

    education = (
        "Less than a Bachelors",
        "Bachelor's degree",
        "Master's degree",
        "Post graduate"
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        person_info = np.array([[country, education, experience]])
        person_info[:, 0] = le_country.transform(person_info[:, 0])
        person_info[:, 1] = le_education.transform(person_info[:, 1])
        person_info = person_info.astype(float)
        
        salary = regressor.predict(person_info)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        st.caption("The above prediction is based on Global Salary Prediction,You can convert it into your country's currency by Multiplying it with the exchange rate of your country.")
        st.subheader('Developed by:')
        st.caption('Priyanka Priyadarshini Dwibedy')
        st.caption('Prachi Pratyasha sahoo')
        st.caption('Biswajit Garnayak')
        st.caption('Satya Narayan')

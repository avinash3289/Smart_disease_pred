import streamlit as st
import joblib
import numpy as np

# Load models
classifier = joblib.load('rf_model.pkl')
heart_classifier = joblib.load('heart_model.pkl')
cancer_model = joblib.load('cancer_model.pkl')

# Define functions for predictions
def predict_diabetes(data):
    return classifier.predict(data)

def predict_heart_disease(data):
    return heart_classifier.predict(data)

def predict_cancer(data):
    return cancer_model.predict(data)

# Define the Streamlit app
def main():
    st.title('Medical Prediction App')

    # Sidebar for navigation
    menu = ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Cancer Prediction']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.write('Welcome to the Medical Prediction App')

    elif choice == 'Diabetes Prediction':
        st.subheader('Diabetes Prediction')
        pregnancies = st.number_input('Pregnancies', value=0)
        glucose = st.number_input('Glucose', value=0)
        blood_pressure = st.number_input('Blood Pressure', value=0)
        skin_thickness = st.number_input('Skin Thickness', value=0)
        insulin = st.number_input('Insulin', value=0)
        bmi = st.number_input('BMI', value=0.0)
        dpf = st.number_input('Diabetes Pedigree Function', value=0.0)
        age = st.number_input('Age', value=0)

        if st.button('Predict'):
            data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            prediction = predict_diabetes(data)
            st.write('Prediction:', prediction)

    elif choice == 'Heart Disease Prediction':
        st.subheader('Heart Disease Prediction')
        # Similar input fields as for Diabetes Prediction...
        if st.button('Predict'):
            # Retrieve input data...
            data = np.array([age, sex, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
                             diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]).reshape(1, -1)
            prediction = predict_heart_disease(data)
            st.write('Prediction:', prediction)

    elif choice == 'Cancer Prediction':
        st.subheader('Cancer Prediction')
        # Similar input fields as for Diabetes Prediction...
        if st.button('Predict'):
            # Retrieve input data...
            data = np.array([radius_mean, texture_mean, smoothness_mean, compactness_mean,
                             symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
                             smoothness_se, compactness_se, concave_points_se, symmetry_se,
                             smoothness_worst, symmetry_worst, fractal_dimension_worst]).reshape(1, -1)
            prediction = predict_cancer(data)
            diagnosis = "Malignant" if prediction == 1 else "Benign"
            st.write('Diagnosis:', diagnosis)

if __name__ == '__main__':
    main()

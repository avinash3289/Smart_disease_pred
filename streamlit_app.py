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

        if st.button('Predict Diabetes'):
            data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            prediction = predict_diabetes(data)
            st.write('Prediction:', prediction)

    elif choice == 'Heart Disease Prediction':
        st.subheader('Heart Disease Prediction')
        sex = int(st.selectbox('Sex', ['Male', 'Female']))
        age = int(st.number_input('Age', value=0))
        education = float(st.number_input('Education', value=0))
        currentSmoker = int(st.selectbox('Current Smoker', ['No', 'Yes']))
        cigsPerDay = float(st.number_input('Cigarettes Per Day', value=0))
        BPMeds = float(st.number_input('BPMeds', value=0))
        prevalentStroke = int(st.selectbox('Prevalent Stroke', ['No', 'Yes']))
        prevalentHyp = int(st.selectbox('Prevalent Hypertension', ['No', 'Yes']))
        diabetes = int(st.selectbox('Diabetes', ['No', 'Yes']))
        totChol = float(st.number_input('Total Cholesterol', value=0))
        sysBP = float(st.number_input('Systolic Blood Pressure', value=0))
        diaBP = float(st.number_input('Diastolic Blood Pressure', value=0))
        BMI = float(st.number_input('BMI', value=0))
        heartRate = float(st.number_input('Heart Rate', value=0))
        glucose = float(st.number_input('Glucose', value=0))

        if st.button('Predict Heart Disease'):
            # Convert input data to appropriate format...
            data = np.array([[sex, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
                              totChol, sysBP, diaBP, BMI, heartRate, glucose]])
            prediction = predict_heart_disease(data)
            st.write('Prediction:', prediction)

    elif choice == 'Cancer Prediction':
        st.subheader('Cancer Prediction')
        radius_mean = st.number_input('Radius Mean', value=0.0)
        texture_mean = st.number_input('Texture Mean', value=0.0)
        smoothness_mean = st.number_input('Smoothness Mean', value=0.0)
        compactness_mean = st.number_input('Compactness Mean', value=0.0)
        symmetry_mean = st.number_input('Symmetry Mean', value=0.0)
        perimeter_mean = st.number_input('Perimeter Mean', value=0.0)
        area_mean = st.number_input('Area Mean', value=0.0)
        concavity_mean = st.number_input('Concavity Mean', value=0.0)
        concave_points_mean = st.number_input('Concave Points Mean', value=0.0)
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean', value=0.0)
        radius_se = st.number_input('Radius SE', value=0.0)
        texture_se = st.number_input('Texture SE', value=0.0)
        smoothness_se = st.number_input('Smoothness SE', value=0.0)
        compactness_se = st.number_input('Compactness SE', value=0.0)
        symmetry_se = st.number_input('Symmetry SE', value=0.0)
        perimeter_se = st.number_input('Perimeter SE', value=0.0)
        area_se = st.number_input('Area SE', value=0.0)
        concavity_se = st.number_input('Concavity SE', value=0.0)
        concave_points_se = st.number_input('Concave Points SE', value=0.0)
        fractal_dimension_se = st.number_input('Fractal Dimension SE', value=0.0)
        radius_worst = st.number_input('Radius Worst', value=0.0)
        texture_worst = st.number_input('Texture Worst', value=0.0)
        smoothness_worst = st.number_input('Smoothness Worst', value=0.0)
        compactness_worst = st.number_input('Compactness Worst', value=0.0)
        symmetry_worst = st.number_input('Symmetry Worst', value=0.0)
        perimeter_worst = st.number_input('Perimeter Worst', value=0.0)
        area_worst = st.number_input('Area Worst', value=0.0)
        concavity_worst = st.number_input('Concavity Worst', value=0.0)
        concave_points_worst = st.number_input('Concave Points Worst', value=0.0)
        fractal_dimension_worst = st.number_input('Fractal Dimension Worst', value=0.0)

        if st.button('Predict Cancer'):
            # Convert input data to appropriate format...
            data = np.array([
    radius_mean, texture_mean, smoothness_mean, compactness_mean, symmetry_mean, 
    perimeter_mean, area_mean, concavity_mean, concave_points_mean, fractal_dimension_mean,
    radius_se, texture_se, smoothness_se, compactness_se, symmetry_se, perimeter_se, 
    area_se, concavity_se, concave_points_se, fractal_dimension_se, radius_worst, 
    texture_worst, smoothness_worst, compactness_worst, symmetry_worst, perimeter_worst, 
    area_worst, concavity_worst, concave_points_worst, fractal_dimension_worst
]).reshape(1, -1)

            prediction = predict_cancer(data)
            diagnosis = "Malignant" if prediction == 1 else "Benign"
            st.write('Diagnosis:', diagnosis)

if __name__ == '__main__':
    main()

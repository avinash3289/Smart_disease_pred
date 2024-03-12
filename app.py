from flask import Flask, render_template, request
import joblib
import numpy as np
filename = 'rf_model.pkl'
classifier = joblib.load(open(filename, 'rb'))
heart_model_filename = 'heart_model.pkl'
try:
    heart_classifier =joblib.load(open(heart_model_filename, 'rb'))
except Exception as e:
    print(f"Error loading heart disease model: {e}")
    heart_classifier = None
cancer_model='cancer_model.pkl'
model=joblib.load(open(cancer_model,'rb'))
app = Flask(__name__,template_folder='Templates')
@app.route('/')
def home():
    return render_template('index.html')  # Remove the leading slash

@app.route('/diabetes', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        return render_template('diabetes.html', prediction=my_prediction)
    return render_template('diabetes.html')


@app.route('/heart', methods=['POST', 'GET'])
def heart():
    if request.method == 'POST':
        sex = int(request.form['sex'])
        age = int(request.form['age'])
        education = float(request.form['education'])
        currentSmoker = int(request.form['currentSmoker'])
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = float(request.form['BPMeds'])
        prevalentStroke = int(request.form['prevalentStroke'])
        prevalentHyp = int(request.form['prevalentHyp'])
        diabetes = int(request.form['diabetes'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])
        
        data = np.array([age, sex, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
                 diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose])
        if data.ndim == 1:
         data = data.reshape(1, -1)  
        my_prediction1 = heart_classifier.predict(data)[0]
        return render_template('result.html', prediction=my_prediction1)

    return render_template('heart.html')
@app.route('/cancer', methods=['GET', 'POST'])
def cancer():
    if request.method == 'POST':
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        smoothness_worst = float(request.form['smoothness_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        input_data = np.array([radius_mean, texture_mean, smoothness_mean, compactness_mean,
                               symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
                               smoothness_se, compactness_se, concave_points_se, symmetry_se,
                               smoothness_worst, symmetry_worst, fractal_dimension_worst]).reshape(1, -1)
        

        
        prediction = model.predict(input_data)
        diagnosis = "Malignant" if prediction == 1 else "Benign"
        return render_template('cancer.html', diagnosis=diagnosis)
    return render_template('cancer.html')

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__, template_folder='templates')

# Replace with your IBM Cloud API key
API_KEY = "nPxtjDFtWSWSnYtJSc5_wVhLCPJgvcb3olyuQ7R7l6Ym"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Fetching access token
            token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
            token_data = token_response.json()
            mltoken = token_data.get("access_token")

            # Check if token retrieval is successful
            if not mltoken:
                raise Exception("Failed to retrieve access token")

            # Define the input data fields and values to be scored
            input_fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            values_to_be_scored = [[int(request.form[field]) for field in input_fields]]

            # Construct the payload for scoring
            payload_scoring = {"input_data": [{"fields": input_fields, "values": values_to_be_scored}]}

            # Send prediction request to IBM Cloud ML
            response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/97d90e7d-e76e-4795-b0a2-3d9eb7ee83bf/predictions?version=2021-05-01', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})

            # Process response and return predictions
            if response_scoring.status_code == 200:
                predictions = response_scoring.json()
                prediction = predictions['predictions'][0]['values'][0][0]
                return render_template('diabetes.html', predictions=prediction)
            else:
                return jsonify({'error': 'Failed to get predictions from IBM Cloud ML'}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('diabetes.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

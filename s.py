import requests
import json


API_KEY = "nPxtjDFtWSWSnYtJSc5_wVhLCPJgvcb3olyuQ7R7l6Ym"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


input_fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
values_to_be_scored = [
    [ 6  ,    148      ,       72     ,        35  ,      0 , 33.6 ,0.627 ,  50  ]    
]

payload_scoring = {"input_data": [{"fields": input_fields, "values": values_to_be_scored}]}
response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/97d90e7d-e76e-4795-b0a2-3d9eb7ee83bf/predictions?version=2021-05-01', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
predictions=response_scoring.json()
print(predictions)
sa=predictions['predictions'][0]['values'][0][0]
if(sa==0):
    print("you don't have diabetes")
else:
    print("you have diabetes..!!")

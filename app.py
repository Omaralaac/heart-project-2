import pickle
from flask_cors import CORS, cross_origin
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open("model1.pkl", "rb"))
@app.route('/')
def home():
    return 'Lets make a prediction'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    json_data = request.json
    print(json_data)

    data = [json_data[key] for key in json_data]
    data_array = np.array(data).reshape(1, -1)
    pred = model.predict(np.array([list(json_data.values())]))
    
    proba = model.predict_proba(data_array)
    proba_of_disease = proba[0][1] 
    
    prop =  round(proba_of_disease * 100, 2)
    #prop["Risk rate"] = str(prop["Risk rate"]) + "%" 


    output = {}
    if int(pred) == 0:
        pred_pat = 'No Heart Disease'
        
    else:
        pred_pat = 'Has Heart Disease'
        
    output = pred_pat , prop

    return jsonify(output)

app.run(host='0.0.0.0', port=8081)

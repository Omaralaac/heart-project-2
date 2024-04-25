import pickle
from flask_cors import CORS, cross_origin
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open("model1.pkl", "rb"))

@app.route('/predict', methods=['POST'])
@cross_origin()
def region():
    json_data = request.json
    print(json_data)

    # تحويل القيم إلى أرقام صحيحة
    for key in json_data:
        json_data[key] = int(json_data[key][0])

    pred = model.predict(np.array([list(json_data.values())]))
    output = {}
    if int(pred) == 0:
        pred_pat = 'No Heart Disease'
    else:
        pred_pat = 'Has Heart Disease'
    output['prediction'] = pred_pat
    return jsonify(output)



app.run(host='0.0.0.0', port=8081)

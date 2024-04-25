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

    pred = model.predict(np.array([list(json_data.values())]))

    # تنسيق الاستجابة
    output = {}
    if int(pred) == 0:
        pred_pat = 'No Heart Disease'
    else:
        pred_pat = 'Has Heart Disease'
    output['prediction'] = pred_pat

    # إرسال الاستجابة كتنسيق JSON
    return jsonify(output)



app.run(host='0.0.0.0', port=8081)

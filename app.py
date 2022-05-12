import json
import torch
import helper
import TTPNet
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

config = json.load(open('./data/config.json', 'r'))

app = Flask(__name__)
CORS(app)

device = torch.device('cpu')
model = TTPNet.TTPNet()
model.load_state_dict(torch.load('./data/chengdu-taxi-sample', map_location=device))
model.eval()

short_ttf = helper.load_file("./data/short_ttf")
long_ttf = helper.load_file("./data/long_ttf")
upstream = helper.load_file("./data/upstream")

border = [102.9, 30.09, 104.9, 31.44]

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with a proper path infomation'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        if json_data is not None:
            coordinates = json_data['coords']
            coords = helper.resample(coordinates)
            distance = json_data['dist']
            start = datetime.now()
            data = helper.transform(coords, distance, start, short_ttf, long_ttf, upstream)
            attr, traj = helper.convert(data)
            pred_dict, loss = model.eval_on_batch(attr, traj, config)
            result = helper.generate(pred_dict, attr)
            return jsonify(result)

if __name__ == '__main__':
    app.debug = True
    app.run()
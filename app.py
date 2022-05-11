# from flask_cors import CORS, cross_origin
import json
import torch
import utils
import random
from model import TTPNet
import collections
import numpy as np
from flask import Flask, jsonify, request
from datetime import datetime

config = json.load(open('./data/config.json', 'r'))

app = Flask(__name__)

device = torch.device('cpu')
model = TTPNet.TTPNet()
model.load_state_dict(torch.load('./data/ttpnet-revise', map_location=device))
model.eval()

short_ttf = utils.load("./data/short_ttf")
long_ttf = utils.load("./data/long_ttf")
upstream = utils.load("./data/upstream")

week = [0]*5 + [1]*2
short_average_speed = 10.72
long_average_speed = 12.36
border = [102.9, 30.09, 104.9, 31.44]

def transform(coords, distance, start):
    data = collections.defaultdict(list)
    data['time'] = distance / (long_average_speed / 3.6)
    data['time_gap'] = [0.0] + [data['time'] // len(coords)] * (len(coords) - 1)
    data['driverID'] = random.randint(1, 10039) - 1
    data['weekID'] = start.weekday()
    data['timeID'] = (start.hour * 60 + start.minute) // 15
    data['dateID'] = week[data['weekID']]
    for i, coord in enumerate(coords):
        data['lngs'].append(coord[0])
        data['lats'].append(coord[1])
        if i == 0:
            dist = 0.0
        else:
            dist += utils.cal_dis(coords[i-1], coord)
        data['dist_gap'].append(dist)
        x, y = utils.map_to_grid(coord)
        data['grid_id'].append(x * 128 + y)
        time_bin = (data['timeID'] + data['time_gap'][i] // (15*60)) % 96
        for j in range(4):
            data['speeds_0'].append(utils.get_speed(short_ttf, x, y, (time_bin-j)%(96)))
        if upstream[x][y]:
            for j in range(4):
                speeds = []
                for k in upstream[x][y]:
                    speeds.append(utils.get_speed(short_ttf, k[0], k[1], (time_bin-j)%(96)))
                data['speeds_1'].append(round(np.mean(speeds), 2))
            max_diff = 100
            for k in upstream[x][y]:
                diff = abs(utils.get_speed(short_ttf, k[0], k[1], time_bin)-utils.get_speed(short_ttf, x, y, time_bin))
                if diff < max_diff and diff != 0:
                    max_diff, kk = diff, k
            for j in range(4):
                data['speeds_2'].append(utils.get_speed(short_ttf, kk[0], kk[1], (time_bin-j)%(96)))
            for j in range(7):
                data['speeds_long'].append(utils.get_speed(long_ttf, x, y, (data['weekID']-j)%7))
        else:
            data['speeds_1'].extend([round(short_average_speed, 2)] * 4)
            data['speeds_2'].extend([round(short_average_speed, 2)] * 4)
        len = 0.0
        if coord != coords[0]:
            xx, yy = utils.map_to_grid(coords[i-1])
            if xx == x and yy == y:
                len += utils.cal_dis(coord, coords[i-1]) / 2
            else:
                len += utils.cal_grid_len([coord[0]-border[0], coord[1]-border[1]], [coords[i-1][0]-border[0], coords[i-1][1]-border[1]], x, y)
        if coord != coords[-1]:
            xx, yy = utils.map_to_grid(coords[i+1])
            if xx == x and yy == y:
                len += utils.cal_dis(coord, coords[i+1]) / 2
            else:
                len += utils.cal_grid_len([coord[0]-border[1], coord[1]-border[1]], [coords[i+1][0]-border[0], coords[i+1][1]-border[1]], x, y)
        data['grid_len'].append(len)
        return data

def convert(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'grid_id', 'time_gap', 'grid_len',
                  'speeds_0', 'speeds_1', 'speeds_2', 'speeds_long']
    attr, traj = {}, {}
    lens = np.asarray([len(data['lngs'])])
    for key in stat_attrs:
        x = torch.FloatTensor([data[key]])
        attr[key] = utils.normalize(x, key)
    for key in info_attrs:
        attr[key] = torch.LongTensor([data[key]])
    for key in traj_attrs:
        if key == 'speeds_0' or key == 'speeds_1' or key == 'speeds_2':
            x = np.asarray([data[key]])
            mask_speeds_forward = np.arange(lens.max()*4) < lens[:, None]*4
            padded = np.zeros(mask_speeds_forward.shape, dtype = np.float32)
            padded[mask_speeds_forward] = np.concatenate(x)
            padded = torch.from_numpy(padded).float()
            padded = padded.reshape(padded.shape[0], -1, 4)
            traj[key] = padded
        elif key == 'speeds_long':
            x = np.asarray([data[key]])
            mask_speeds_history = np.arange(lens.max()*7) < lens[:, None]*7
            padded = np.zeros(mask_speeds_history.shape, dtype = np.float32)
            padded[mask_speeds_history] = np.concatenate(x)
            padded = torch.from_numpy(padded).float()
            padded = padded.reshape(padded.shape[0], -1, 7)
            traj[key] = padded
        elif key == 'grid_id':
            x = np.asarray([data[key]])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            padded = torch.LongTensor(padded)
            traj[key] = padded
        elif key == 'time_gap':
            x = np.asarray([data[key]])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.ones(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            T_f = torch.from_numpy(padded).float()
            T_f = T_f[:, 1:]
            mask_f = mask[:, 1:]
            M_f = np.zeros(mask_f.shape, dtype = np.int)
            M_f[mask_f] = 1
            M_f = torch.from_numpy(M_f).float()
            traj['T_f'] = T_f
            traj['M_f'] = M_f
        elif key == 'grid_len':
            x = np.asarray([data[key]])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            padded = torch.from_numpy(padded).float()
            traj[key] = padded
        else:
            x = np.asarray([data[key]])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            padded = utils.normalize(padded, key)
            padded = torch.from_numpy(padded).float()
            traj[key] = padded
    lens = lens.tolist()
    traj['lens'] = lens
    attr, traj = utils.to_var(attr), utils.to_var(traj)
    return attr, traj

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with a proper path infomation'})

@app.route('/predict', methods=['POST'])
# @cross_origin()
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        if json_data is not None:
            coordinates = json_data['coordinates']
            coords = utils.resample(coordinates)
            distance = json_data['distance']
            start = datetime.now()
            data = transform(coords, distance, start)
            attr, traj = convert(data)
            pred_dict = model.eval_on_batch(attr, traj, config)
            pred = pred_dict['pred'].data.cpu().numpy()
            return jsonify({ 'predict_time' : pred[0][0] })

if __name__ == '__main__':
    app.run()
import json
import math
import torch
import random
import collections
import numpy as np
import utils

week = [0]*5 + [1]*2
short_average_speed = 10.72
long_average_speed = 12.36
border = [102.9, 30.09, 104.9, 31.44]
grid_size = [(border[2] - border[0]) / 128, (border[3] - border[1]) / 128]
config = json.load(open('./data/config.json', 'r'))

def load_file(name):
    arr = []
    with open(name, "r") as file:
        lines = file.readlines()
        for line in lines:
            arr.append(json.loads(line))
    return arr

def cal_dis(p1, p2):
    R = 6370
    lon1 = math.radians(p1[0])
    lat1 = math.radians(p1[1])
    lon2 = math.radians(p2[0])
    lat2 = math.radians(p2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(R * c, 10)

def map_to_grid(coord):
    x = int((coord[0] - border[0]) // grid_size[0])
    y = int((coord[1] - border[1]) // grid_size[1])
    return x, y

def get_speed(ttf, x, y, ind):
    try:
        speed = ttf[x][y][ind][0]
    except (KeyError, IndexError):
        speed = ttf[x][y]["-1"][0]
    return round(speed * 3.6, 2)

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.a = (y2 - y1) / (x2 - x1)
        self.b = y1 - self.a * x1

    def y_to_x(self, y):
        return (y - self.b) / self.a

    def x_to_y(self, x):
        return self.a * x + self.b

def cal_grid_len(p1, p2, x, y):
    line1, line2 = x * grid_size[0], (x + 1) * grid_size[0]
    line3, line4 = y * grid_size[1], (y + 1) * grid_size[1]
    if p1[0] == p2[0]:
        if min(p1[1], p2[1]) <= line3 <= max(p1[1], p2[1]):
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [line3+border[0], p1[1]+border[1]])
        else:
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [line4+border[0], p1[1]+border[1]])
    elif p1[1] == p2[1]:
        if min(p1[0], p2[0]) <= line1 <= max(p1[0], p2[0]):
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [p1[0]+border[0], line1+border[1]])
        else:
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [p1[0]+border[0], line2+border[1]])
    else:
        line = Line(p1[0], p1[1], p2[0], p2[1])
        if line1 < line.y_to_x(line3) < line2 and min(p1[1], p2[1]) <= line3 <= max(p1[1], p2[1]):
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [line.y_to_x(line3)+border[0], line3+border[1]])
        elif line1 < line.y_to_x(line4) < line2 and min(p1[1], p2[1]) <= line4 <= max(p2[1], p2[1]):
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [line.y_to_x(line4)+border[0], line4+border[1]])
        elif line3 < line.x_to_y(line1) < line4 and min(p1[0], p2[0]) <= line1 <= max(p1[0], p2[0]):
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [line1+border[0], line.x_to_y(line1)+border[1]])
        else:
            return cal_dis([p1[0]+border[0], p1[1]+border[1]], [line2+border[0], line.x_to_y(line2)+border[1]])

def resample(coordinates):
    coords = []
    for i, coordinate in enumerate(coordinates):
        if i % 5 == 0:
            coords.append(coordinate)
    return coords

def transform(coords, distance, start, short_ttf, long_ttf, upstream):
    query = collections.defaultdict(lambda: [])
    query['driverID'] = random.randint(1, 10039) - 1
    query['weekID'] = start.weekday()
    query['timeID'] = (start.hour * 60 + start.minute) // 15
    query['dateID'] = week[query['weekID']]
    query['dist'] = distance
    query['time'] = distance / (long_average_speed / 3.6)
    query['time_gap'] = [0.0] + [query['time'] // (len(coords)-1)] * (len(coords) - 1)
    for i, coord in enumerate(coords):
        query['lngs'].append(coord[0])
        query['lats'].append(coord[1])
        if i == 0:
            dist = 0.0
        else:
            dist += cal_dis(coords[i-1], coord)
        query['dist_gap'].append(dist)
        x, y = map_to_grid(coord)
        query['grid_id'].append(x * 128 + y)
        time_bin = (query['timeID'] + query['time_gap'][i] // (15*60)) % 96
        for j in range(4):
            query['speeds_0'].append(get_speed(short_ttf, x, y, (time_bin-j)%(96)))
        if upstream[x][y]:
            for j in range(4):
                speeds = []
                for k in upstream[x][y]:
                    speeds.append(get_speed(short_ttf, k[0], k[1], (time_bin-j)%(96)))
                query['speeds_1'].append(round(np.mean(speeds), 2))
            max_diff = 100
            for k in upstream[x][y]:
                diff = abs(get_speed(short_ttf, k[0], k[1], time_bin)-get_speed(short_ttf, x, y, time_bin))
                if diff < max_diff and diff != 0:
                    max_diff, kk = diff, k
            for j in range(4):
                query['speeds_2'].append(get_speed(short_ttf, kk[0], kk[1], (time_bin-j)%(96)))
            for j in range(7):
                query['speeds_long'].append(get_speed(long_ttf, x, y, (query['weekID']-j)%7))
        else:
            query['speeds_1'].extend([round(short_average_speed, 2)] * 4)
            query['speeds_2'].extend([round(short_average_speed, 2)] * 4)
        length = 0.0
        if coord != coords[0]:
            xx, yy = map_to_grid(coords[i-1])
            if xx == x and yy == y:
                length += cal_dis(coord, coords[i-1]) / 2
            else:
                length += cal_grid_len([coord[0]-border[0], coord[1]-border[1]], [coords[i-1][0]-border[0], coords[i-1][1]-border[1]], x, y)
        if coord != coords[-1]:
            xx, yy = map_to_grid(coords[i+1])
            if xx == x and yy == y:
                length += cal_dis(coord, coords[i+1]) / 2
            else:
                length += cal_grid_len([coord[0]-border[0], coord[1]-border[1]], [coords[i+1][0]-border[0], coords[i+1][1]-border[1]], x, y)
        query['grid_len'].append(length)
    data = []
    for i in range(32):
        data.append(query)
    return data

def convert(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']

    traj_attrs = ['lngs', 'lats', 'grid_id', 'time_gap', 'grid_len',
                  'speeds_0', 'speeds_1', 'speeds_2', 'speeds_long']
    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])
    
    for key in traj_attrs:
        if key == 'speeds_0' or key == 'speeds_1' or key == 'speeds_2':
            x = np.asarray([item[key] for item in data])
            mask_speeds_forward = np.arange(lens.max()*4) < lens[:, None]*4
            padded = np.zeros(mask_speeds_forward.shape, dtype = np.float32)
            padded[mask_speeds_forward] = np.concatenate(x)
            
            padded = torch.from_numpy(padded).float()
            padded = padded.reshape(padded.shape[0], -1, 4)
            traj[key] = padded
        
        elif key == 'speeds_long':
            x = np.asarray([item[key] for item in data])
            mask_speeds_history = np.arange(lens.max()*7) < lens[:, None]*7
            padded = np.zeros(mask_speeds_history.shape, dtype = np.float32)
            padded[mask_speeds_history] = np.concatenate(x)
            
            padded = torch.from_numpy(padded).float()
            padded = padded.reshape(padded.shape[0], -1, 7)
            traj[key] = padded
            
        elif key == 'grid_id':
            x = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            
            padded = torch.LongTensor(padded)
            traj[key] = padded

        elif key == 'time_gap':
            x = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.ones(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            
            # label
            T_f = torch.from_numpy(padded).float()
            T_f = T_f[:, 1:]
            mask_f = mask[:, 1:]
            M_f = np.zeros(mask_f.shape, dtype = np.int)
            M_f[mask_f] = 1
            M_f = torch.from_numpy(M_f).float()
            
            traj['T_f'] = T_f
            traj['M_f'] = M_f
        
        elif key == 'grid_len':
            x = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            
            padded = torch.from_numpy(padded).float()
            traj[key] = padded
            
        else:
            x = np.asarray([item[key] for item in data])
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

def generate(pred_dict, attr):
    dist_list, time_list = [], []
    pred = pred_dict['pred'].data.cpu().numpy()
    for i in range(pred_dict['pred'].size()[0]):
        dist_list.append(utils.unnormalize(attr['dist'].data[i], 'dist'))
        time_list.append(pred[i][0])
    time = np.mean(time_list)
    result = {
        'time': str(time)
    }
    return result
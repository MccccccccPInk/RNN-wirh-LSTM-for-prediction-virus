import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def creat_dataset(dataset, look_back, look_forward):
    data_x = []
    data_y = []
    for i in range(len(dataset)-look_back-look_forward):
        data_x.append(dataset[i:i+look_back])
        data_y.append(dataset[i+look_back: i+look_back+look_forward])
    return np.asarray(data_x), np.asarray(data_y)


def build_set(name_list: list, look_back=21, look_forward=21):
    print("constructing", name_list)
    set_X = np.ndarray([])
    set_Y = np.ndarray([])
    for region_name in name_list:
        print('adding', region_name)
        set_df = pd.read_csv('./data/data_processed/train/' + region_name + '.csv')
        set_np = set_df[['N', 'susceptible', 'existing infected', 'cured', 'dead']].values.astype(int)

        # Fit the scaler to the training set and standardize
        scaler = MinMaxScaler(feature_range=[0, 1]).fit(set_np)
        # set_np = scaler.transform(set_np)
        # set_np = set_np + 1

        dataX, dataY = creat_dataset(set_np, look_back, look_forward)

        set_X = np.append(set_X, dataX)
        set_Y = np.append(set_Y, dataY)

    set_X = np.delete(set_X, 0)
    set_Y = np.delete(set_Y, 0)

    set_X = np.reshape(set_X, [-1, look_back, 5])
    set_Y = np.reshape(set_Y, [-1, look_forward, 5])

    return set_X, set_Y, scaler


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

from printErr import printErr


def plot_pred(model_used, region_name_list, scaler_input):
    for region_name in region_name_list:
        print("predicting" + region_name)
        data_df = pd.read_csv(filepath_or_buffer='./data/data_processed/test/' + region_name + '.csv',
                              usecols=['date', 'N', 'susceptible', 'existing infected', 'cured', 'dead'])
        data_np = data_df[['N', 'susceptible', 'existing infected', 'cured', 'dead']].values.astype(int)
        # normalize using the input scaler
        # data_np = scaler_input.transform(data_np)
        # data_np = data_np + 1

        plt.axvline(x=147, color='r', linestyle='--')
        plt.axvline(x=168, color='r', linestyle='--')

        # the first prediction
        seed = data_np[147:168, :].reshape(1, 21, 5)
        y_hat = model_used.predict(seed)
        pred_np = y_hat
        for i in range(0, 4 - 1):
            y_hat = model_used.predict(y_hat)
            pred_np = np.append(pred_np, y_hat)

        pred_np = pred_np.reshape(-1, 5)
        plt.plot(np.arange(168, 252, 1), pred_np[:, 2], label='pred_I', linestyle="--")

        plt.title(region_name)
        plt.plot(data_np[:, 2], label='I')
        # plt.plot(data_np[:, 3], label='R')
        # plt.plot(data_np[:, 4], label='D')

        x_major_locator = MultipleLocator(50)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.legend()
        plt.xlabel('day')
        plt.savefig('./plot/test/' + region_name + '.jpg')
        plt.show()

        print("I:")
        printErr(data_np[168:, 2], pred_np[:, 2])

        print("R:")
        printErr(data_np[168:, 3], pred_np[:, 3])

        print("D:")
        printErr(data_np[168:, 4], pred_np[:, 4])

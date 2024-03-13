from sklearn.metrics import mean_squared_error #mean squared error
from sklearn.metrics import mean_absolute_error


def printErr(y, y_predict):
    print("MAE: ", mean_absolute_error(y, y_predict))
    print("MSE: ", mean_squared_error(y, y_predict))

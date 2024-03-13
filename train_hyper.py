import matplotlib.pyplot as plt
import tensorflow as tf

from dataset_maker import build_set
from model_builder import build_model
from loss import WeightedMAE, TimeWeightedMAE
from plot import plot_pred


train_list = [['ShangHai', 'ShanDong', 'YunNan', 'BeiJing', 'TianJin', 'HeBei', 'LiaoNing', 'JiLin', 'HeiLongJiang',
             'JiangSu', 'ZheJiang', 'AnHui', 'JiangXi', 'HuNan', 'GuangDong', 'GuangXi', 'HaiNan', 'ChongQin',
              'SiChuan', 'GuiZhou', 'GanSu', 'NeiMeng', 'FuJian', 'NingXia', 'QingHai', 'ShanXi']]

for region_name in train_list:
    train_X, train_Y, scaler = build_set(region_name)

    model = build_model()
    loss = TimeWeightedMAE()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=8, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.00001, batch_size=8)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=8, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.0001, batch_size=8)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=8, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.001, batch_size=8)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=8, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.01, batch_size=8)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=16, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.00001, batch_size=16)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=16, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.0001, batch_size=16)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=16, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.001, batch_size=16)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=16, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.01, batch_size=16)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=32, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.00001, batch_size=32)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=32, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.0001, batch_size=32)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=32, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.001, batch_size=32)')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=32, shuffle=True, verbose=1)
    # visualize the loss
    plt.plot(history.history['loss'], label='loss(lr=0.01, batch_size=32)')

    plt.title("training process")
    plt.legend()
    plt.show()

print("over!!!!")

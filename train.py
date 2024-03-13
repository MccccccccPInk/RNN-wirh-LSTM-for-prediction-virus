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
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    # training the model
    print("training")
    history = model.fit(train_X, train_Y, epochs=10, batch_size=8, shuffle=True, verbose=1)
    # visualize the loss
    plt.title("training process")
    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.show()

    plot_pred(model_used=model,
              region_name_list=region_name,
              scaler_input=scaler)

print("over!!!!!!!!!!!!！！")

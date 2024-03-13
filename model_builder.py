import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Concatenate, Reshape, Flatten
from SEIRLayer import SEIRLayer

def build_model():
    X_1_21 = tf.keras.Input(shape=(21, 5))

    X_2 = tf.gather(X_1_21, axis=1, indices=[1])
    X_3 = tf.gather(X_1_21, axis=1, indices=[2])
    X_4 = tf.gather(X_1_21, axis=1, indices=[3])
    X_5 = tf.gather(X_1_21, axis=1, indices=[4])
    X_6 = tf.gather(X_1_21, axis=1, indices=[5])
    X_7 = tf.gather(X_1_21, axis=1, indices=[6])

    X_8 = tf.gather(X_1_21, axis=1, indices=[7])
    X_9 = tf.gather(X_1_21, axis=1, indices=[8])
    X_10 = tf.gather(X_1_21, axis=1, indices=[9])
    X_11 = tf.gather(X_1_21, axis=1, indices=[10])
    X_12 = tf.gather(X_1_21, axis=1, indices=[11])
    X_13 = tf.gather(X_1_21, axis=1, indices=[12])
    X_14 = tf.gather(X_1_21, axis=1, indices=[13])

    X_15 = tf.gather(X_1_21, axis=1, indices=[14])
    X_16 = tf.gather(X_1_21, axis=1, indices=[15])
    X_17 = tf.gather(X_1_21, axis=1, indices=[16])
    X_18 = tf.gather(X_1_21, axis=1, indices=[17])
    X_19 = tf.gather(X_1_21, axis=1, indices=[18])
    X_20 = tf.gather(X_1_21, axis=1, indices=[19])
    X_21 = tf.gather(X_1_21, axis=1, indices=[20])

    lstm_layer1 = LSTM(128, recurrent_activation='sigmoid', activation='tanh', dropout=0.1, return_sequences=True,
                       input_shape=(21, 3))
    lstm_layer2 = LSTM(128, recurrent_activation='sigmoid', activation='tanh', dropout=0.1, return_sequences=False,
                       input_shape=(21, 128))
    FC_layer = Dense(6, activation='softplus')
    SEIR = SEIRLayer()

    IRD = tf.gather(X_1_21, axis=2, indices=[2, 3, 4])
    LSTM1_out_1 = lstm_layer1(IRD)
    LSTM2_out_1 = lstm_layer2(LSTM1_out_1)
    FC_out_1 = FC_layer(LSTM2_out_1)
    three_days_data_1 = Reshape([105])(X_1_21)
    x1 = Concatenate(axis=-1)([three_days_data_1, FC_out_1])
    Y_1 = SEIR(x1)
    Y_1 = Reshape([1, 5])(Y_1)
    X_2_21_Y_1 = Concatenate(axis=1)([X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14,
                                      X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                      Y_1])

    IRD_2 = tf.gather(X_2_21_Y_1, axis=2, indices=[2, 3, 4])
    LSTM1_out_2 = lstm_layer1(IRD_2)
    LSTM2_out_2 = lstm_layer2(LSTM1_out_2)
    FC_out_2 = FC_layer(LSTM2_out_2)
    three_days_data_2 = Reshape([105])(X_2_21_Y_1)
    x2 = Concatenate(axis=-1)([three_days_data_2, FC_out_2])
    Y_2 = SEIR(x2)
    Y_2 = Reshape([1, 5])(Y_2)
    X_3_21_Y_1_2 = Concatenate(axis=1)([X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14,
                                        X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                        Y_1, Y_2])

    IRD_3 = tf.gather(X_3_21_Y_1_2, axis=2, indices=[2, 3, 4])
    LSTM1_out_3 = lstm_layer1(IRD_3)
    LSTM2_out_3 = lstm_layer2(LSTM1_out_3)
    FC_out_3 = FC_layer(LSTM2_out_3)
    three_days_data_3 = Reshape([105])(X_3_21_Y_1_2)
    x3 = Concatenate(axis=-1)([three_days_data_3, FC_out_3])
    Y_3 = SEIR(x3)
    Y_3 = Reshape([1, 5])(Y_3)
    X_4_21_Y_1_3 = Concatenate(axis=1)([X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14,
                                        X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                        Y_1, Y_2, Y_3])

    IRD_4 = tf.gather(X_4_21_Y_1_3, axis=2, indices=[2, 3, 4])
    LSTM1_out_4 = lstm_layer1(IRD_4)
    LSTM2_out_4 = lstm_layer2(LSTM1_out_4)
    FC_out_4 = FC_layer(LSTM2_out_4)
    three_days_data_4 = Reshape([105])(X_4_21_Y_1_3)
    x4 = Concatenate(axis=-1)([three_days_data_4, FC_out_4])
    Y_4 = SEIR(x4)
    Y_4 = Reshape([1, 5])(Y_4)
    X_5_21_Y_1_4 = Concatenate(axis=1)([X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14,
                                        X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                        Y_1, Y_2, Y_3, Y_4])

    IRD_5 = tf.gather(X_5_21_Y_1_4, axis=2, indices=[2, 3, 4])
    LSTM1_out_5 = lstm_layer1(IRD_5)
    LSTM2_out_5 = lstm_layer2(LSTM1_out_5)
    FC_out_5 = FC_layer(LSTM2_out_5)
    three_days_data_5 = Reshape([105])(X_5_21_Y_1_4)
    x5 = Concatenate(axis=-1)([three_days_data_5, FC_out_5])
    Y_5 = SEIR(x5)
    Y_5 = Reshape([1, 5])(Y_5)
    X_6_21_Y_1_5 = Concatenate(axis=1)([X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14,
                                        X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                        Y_1, Y_2, Y_3, Y_4, Y_5])

    IRD_6 = tf.gather(X_6_21_Y_1_5, axis=2, indices=[2, 3, 4])
    LSTM1_out_6 = lstm_layer1(IRD_6)
    LSTM2_out_6 = lstm_layer2(LSTM1_out_6)
    FC_out_6 = FC_layer(LSTM2_out_6)
    three_days_data_6 = Reshape([105])(X_6_21_Y_1_5)
    x6 = Concatenate(axis=-1)([three_days_data_6, FC_out_6])
    Y_6 = SEIR(x6)
    Y_6 = Reshape([1, 5])(Y_6)
    X_7_21_Y_1_6 = Concatenate(axis=1)([X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14,
                                        X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                        Y_1, Y_2, Y_3, Y_4, Y_5, Y_6])

    IRD_7 = tf.gather(X_7_21_Y_1_6, axis=2, indices=[2, 3, 4])
    LSTM1_out_7 = lstm_layer1(IRD_7)
    LSTM2_out_7 = lstm_layer2(LSTM1_out_7)
    FC_out_7 = FC_layer(LSTM2_out_7)
    three_days_data_7 = Reshape([105])(X_7_21_Y_1_6)
    x7 = Concatenate(axis=-1)([three_days_data_7, FC_out_7])
    Y_7 = SEIR(x7)
    Y_7 = Reshape([1, 5])(Y_7)
    X_8_21_Y_1_7 = Concatenate(axis=1)([X_8, X_9, X_10, X_11, X_12, X_13, X_14,
                                        X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                        Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7])

    IRD_8 = tf.gather(X_8_21_Y_1_7, axis=2, indices=[2, 3, 4])
    LSTM1_out_8 = lstm_layer1(IRD_8)
    LSTM2_out_8 = lstm_layer2(LSTM1_out_8)
    FC_out_8 = FC_layer(LSTM2_out_8)
    three_days_data_8 = Reshape([105])(X_8_21_Y_1_7)
    x8 = Concatenate(axis=-1)([three_days_data_8, FC_out_8])
    Y_8 = SEIR(x8)
    Y_8 = Reshape([1, 5])(Y_8)
    X_9_21_Y_1_8 = Concatenate(axis=1)([X_9, X_10, X_11, X_12, X_13, X_14,
                                        X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                        Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8])

    IRD_9 = tf.gather(X_9_21_Y_1_8, axis=2, indices=[2, 3, 4])
    LSTM1_out_9 = lstm_layer1(IRD_9)
    LSTM2_out_9 = lstm_layer2(LSTM1_out_9)
    FC_out_9 = FC_layer(LSTM2_out_9)
    three_days_data_9 = Reshape([105])(X_9_21_Y_1_8)
    x9 = Concatenate(axis=-1)([three_days_data_9, FC_out_9])
    Y_9 = SEIR(x9)
    Y_9 = Reshape([1, 5])(Y_9)
    X_10_21_Y_1_9 = Concatenate(axis=1)([X_10, X_11, X_12, X_13, X_14,
                                         X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                         Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9])

    IRD_10 = tf.gather(X_10_21_Y_1_9, axis=2, indices=[2, 3, 4])
    LSTM1_out_10 = lstm_layer1(IRD_10)
    LSTM2_out_10 = lstm_layer2(LSTM1_out_10)
    FC_out_10 = FC_layer(LSTM2_out_10)
    three_days_data_10 = Reshape([105])(X_10_21_Y_1_9)
    x10 = Concatenate(axis=-1)([three_days_data_10, FC_out_10])
    Y_10 = SEIR(x10)
    Y_10 = Reshape([1, 5])(Y_10)
    X_11_21_Y_1_10 = Concatenate(axis=1)([X_11, X_12, X_13, X_14,
                                          X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10])

    IRD_11 = tf.gather(X_11_21_Y_1_10, axis=2, indices=[2, 3, 4])
    LSTM1_out_11 = lstm_layer1(IRD_11)
    LSTM2_out_11 = lstm_layer2(LSTM1_out_11)
    FC_out_11 = FC_layer(LSTM2_out_11)
    three_days_data_11 = Reshape([105])(X_11_21_Y_1_10)
    x11 = Concatenate(axis=-1)([three_days_data_11, FC_out_11])
    Y_11 = SEIR(x11)
    Y_11 = Reshape([1, 5])(Y_11)
    X_12_21_Y_1_11 = Concatenate(axis=1)([X_12, X_13, X_14,
                                          X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11])

    IRD_12 = tf.gather(X_12_21_Y_1_11, axis=2, indices=[2, 3, 4])
    LSTM1_out_12 = lstm_layer1(IRD_12)
    LSTM2_out_12 = lstm_layer2(LSTM1_out_12)
    FC_out_12 = FC_layer(LSTM2_out_12)
    three_days_data_12 = Reshape([105])(X_12_21_Y_1_11)
    x12 = Concatenate(axis=-1)([three_days_data_12, FC_out_12])
    Y_12 = SEIR(x12)
    Y_12 = Reshape([1, 5])(Y_12)
    X_13_21_Y_1_12 = Concatenate(axis=1)([X_13, X_14,
                                          X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12])

    IRD_13 = tf.gather(X_13_21_Y_1_12, axis=2, indices=[2, 3, 4])
    LSTM1_out_13 = lstm_layer1(IRD_13)
    LSTM2_out_13 = lstm_layer2(LSTM1_out_13)
    FC_out_13 = FC_layer(LSTM2_out_13)
    three_days_data_13 = Reshape([105])(X_13_21_Y_1_12)
    x13 = Concatenate(axis=-1)([three_days_data_13, FC_out_13])
    Y_13 = SEIR(x13)
    Y_13 = Reshape([1, 5])(Y_13)
    X_14_21_Y_1_13 = Concatenate(axis=1)([X_14,
                                          X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13])

    IRD_14 = tf.gather(X_14_21_Y_1_13, axis=2, indices=[2, 3, 4])
    LSTM1_out_14 = lstm_layer1(IRD_14)
    LSTM2_out_14 = lstm_layer2(LSTM1_out_14)
    FC_out_14 = FC_layer(LSTM2_out_14)
    three_days_data_14 = Reshape([105])(X_14_21_Y_1_13)
    x14 = Concatenate(axis=-1)([three_days_data_14, FC_out_14])
    Y_14 = SEIR(x14)
    Y_14 = Reshape([1, 5])(Y_14)
    X_15_21_Y_1_14 = Concatenate(axis=1)([X_15, X_16, X_17, X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14])

    IRD_15 = tf.gather(X_15_21_Y_1_14, axis=2, indices=[2, 3, 4])
    LSTM1_out_15 = lstm_layer1(IRD_15)
    LSTM2_out_15 = lstm_layer2(LSTM1_out_15)
    FC_out_15 = FC_layer(LSTM2_out_15)
    three_days_data_15 = Reshape([105])(X_15_21_Y_1_14)
    x15 = Concatenate(axis=-1)([three_days_data_15, FC_out_15])
    Y_15 = SEIR(x15)
    Y_15 = Reshape([1, 5])(Y_15)
    X_16_21_Y_1_15 = Concatenate(axis=1)([X_16, X_17, X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14,
                                          Y_15])

    IRD_16 = tf.gather(X_16_21_Y_1_15, axis=2, indices=[2, 3, 4])
    LSTM1_out_16 = lstm_layer1(IRD_16)
    LSTM2_out_16 = lstm_layer2(LSTM1_out_16)
    FC_out_16 = FC_layer(LSTM2_out_16)
    three_days_data_16 = Reshape([105])(X_16_21_Y_1_15)
    x16 = Concatenate(axis=-1)([three_days_data_16, FC_out_16])
    Y_16 = SEIR(x16)
    Y_16 = Reshape([1, 5])(Y_16)
    X_17_21_Y_1_16 = Concatenate(axis=1)([X_17, X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14,
                                          Y_15, Y_16])

    IRD_17 = tf.gather(X_17_21_Y_1_16, axis=2, indices=[2, 3, 4])
    LSTM1_out_17 = lstm_layer1(IRD_17)
    LSTM2_out_17 = lstm_layer2(LSTM1_out_17)
    FC_out_17 = FC_layer(LSTM2_out_17)
    three_days_data_17 = Reshape([105])(X_17_21_Y_1_16)
    x17 = Concatenate(axis=-1)([three_days_data_17, FC_out_17])
    Y_17 = SEIR(x17)
    Y_17 = Reshape([1, 5])(Y_17)
    X_18_21_Y_1_17 = Concatenate(axis=1)([X_18, X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14,
                                          Y_15, Y_16, Y_17])

    IRD_18 = tf.gather(X_18_21_Y_1_17, axis=2, indices=[2, 3, 4])
    LSTM1_out_18 = lstm_layer1(IRD_18)
    LSTM2_out_18 = lstm_layer2(LSTM1_out_18)
    FC_out_18 = FC_layer(LSTM2_out_18)
    three_days_data_18 = Reshape([105])(X_18_21_Y_1_17)
    x18 = Concatenate(axis=-1)([three_days_data_18, FC_out_18])
    Y_18 = SEIR(x18)
    Y_18 = Reshape([1, 5])(Y_18)
    X_19_21_Y_1_18 = Concatenate(axis=1)([X_19, X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14,
                                          Y_15, Y_16, Y_17, Y_18])

    IRD_19 = tf.gather(X_19_21_Y_1_18, axis=2, indices=[2, 3, 4])
    LSTM1_out_19 = lstm_layer1(IRD_19)
    LSTM2_out_19 = lstm_layer2(LSTM1_out_19)
    FC_out_19 = FC_layer(LSTM2_out_19)
    three_days_data_19 = Reshape([105])(X_19_21_Y_1_18)
    x19 = Concatenate(axis=-1)([three_days_data_19, FC_out_19])
    Y_19 = SEIR(x19)
    Y_19 = Reshape([1, 5])(Y_19)
    X_20_21_Y_1_19 = Concatenate(axis=1)([X_20, X_21,
                                          Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14,
                                          Y_15, Y_16, Y_17, Y_18, Y_19])

    IRD_20 = tf.gather(X_20_21_Y_1_19, axis=2, indices=[2, 3, 4])
    LSTM1_out_20 = lstm_layer1(IRD_20)
    LSTM2_out_20 = lstm_layer2(LSTM1_out_20)
    FC_out_20 = FC_layer(LSTM2_out_20)
    three_days_data_20 = Reshape([105])(X_20_21_Y_1_19)
    x20 = Concatenate(axis=-1)([three_days_data_20, FC_out_20])
    Y_20 = SEIR(x20)
    Y_20 = Reshape([1, 5])(Y_20)
    X_21_Y_1_20 = Concatenate(axis=1)([X_21,
                                       Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14,
                                       Y_15, Y_16, Y_17, Y_18, Y_19, Y_20])

    IRD_21 = tf.gather(X_21_Y_1_20, axis=2, indices=[2, 3, 4])
    LSTM1_out_21 = lstm_layer1(IRD_21)
    LSTM2_out_21 = lstm_layer2(LSTM1_out_21)
    FC_out_21 = FC_layer(LSTM2_out_21)
    three_days_data_21 = Reshape([105])(X_21_Y_1_20)
    x21 = Concatenate(axis=-1)([three_days_data_21, FC_out_21])
    Y_21 = SEIR(x21)
    Y_21 = Reshape([1, 5])(Y_21)
    Y_1_21 = Concatenate(axis=1)([Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8, Y_9, Y_10, Y_11, Y_12, Y_13, Y_14,
                                  Y_15, Y_16, Y_17, Y_18, Y_19, Y_20, Y_21])

    Net1_output = Y_1_21

    model = tf.keras.Model(inputs=X_1_21, outputs=Net1_output)
    # print(model.summary())

    return model

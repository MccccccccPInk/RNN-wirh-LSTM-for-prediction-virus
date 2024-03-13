import tensorflow as tf
from tensorflow.math import exp


class SEIRLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        print("call SEIRLayer Init!")

    def build(self, input_shape):
        print("call SEIRLayer Build!")

    def call(self, inputs):
        beta = tf.gather(inputs, axis=1, indices=[105])
        beta_b = tf.gather(inputs, axis=1, indices=[106])
        gamma = tf.gather(inputs, axis=1, indices=[107])
        gamma_b = tf.gather(inputs, axis=1, indices=[108])
        mu = tf.gather(inputs, axis=1, indices=[109])
        mu_b = tf.gather(inputs, axis=1, indices=[110])

        # beta = beta_a * (exp(-1 / beta_b))
        # gamma = gamma_a * (exp(-1/gamma_b))
        # mu = mu_a * (exp(-1/mu_b))

        N = tf.gather(inputs, axis=1, indices=[100])

        S_t = tf.gather(inputs, axis=1, indices=[101])
        I_t = tf.gather(inputs, axis=1, indices=[102])
        R_t = tf.gather(inputs, axis=1, indices=[103])
        D_t = tf.gather(inputs, axis=1, indices=[104])

        # S_total = S_t
        # I_total = I_t
        #
        # for i in range(0, 20):
        #     S_total += tf.gather(inputs, axis=1, indices=[5 * i + 1])
        #     I_total += tf.gather(inputs, axis=1, indices=[5 * i + 2])
        #
        # S_avg = S_total / 21
        # I_avg = I_total / 21

        # S_max = S_t
        I_max = I_t

        for i in range(0, 20):
            # S_tmp = tf.gather(inputs, axis=1, indices=[5 * i + 1])
            # if S_tmp[0][0] > S_max[0][0]:
            #     S_max = S_tmp
            I_tmp = tf.gather(inputs, axis=1, indices=[5 * i + 2])
            if I_tmp[0][0] > I_max[0][0]:
                I_max = I_tmp

        S_pred = S_t - beta * I_max * S_t / N
        I_pred = I_t + beta * I_max * S_t / N - gamma * I_max - mu * I_max
        R_pred = R_t + gamma * I_max
        D_pred = D_t + mu * I_max

        pred = tf.concat([N, S_pred, I_pred, R_pred, D_pred], 1)

        return pred

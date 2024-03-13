from SEIRLayer import SEIRLayer
import tensorflow as tf

test_SEIRLayer = SEIRLayer()

a = tf.constant([[1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  1, 2, 3, 4, 5,
                 3000045, 3000000, 15, 10, 20,  # N,S,I,R,D for today
                 1.1, 0, 0, 1.1, 0, 0]], dtype=tf.double)  # FC_out
b = test_SEIRLayer(a)

print(b)



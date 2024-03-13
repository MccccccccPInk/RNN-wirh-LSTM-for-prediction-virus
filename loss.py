import tensorflow as tf


class WeightedMAE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        I = tf.gather(y_true, axis=2, indices=[2])
        R = tf.gather(y_true, axis=2, indices=[3])
        D = tf.gather(y_true, axis=2, indices=[4])

        I_pred = tf.gather(y_pred, axis=2, indices=[2])
        R_pred = tf.gather(y_pred, axis=2, indices=[3])
        D_pred = tf.gather(y_pred, axis=2, indices=[4])

        # loss = tf.reduce_mean(tf.square(I_pred - I)) + \
        #        tf.reduce_mean(tf.square(R_pred - R)) + \
        #        tf.reduce_mean(tf.square(D_pred - D))
        loss = 0.4 * tf.reduce_mean(tf.abs(I_pred - I)) + \
               0.2 * tf.reduce_mean(tf.abs(R_pred - R)) + \
               0.4 * tf.reduce_mean(tf.abs(D_pred - D))
        # loss = tf.reduce_mean(tf.abs(I_pred - I))
        return loss


class TimeWeightedMAE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        day1 = tf.gather(y_true, axis=1, indices=[0])
        day2 = tf.gather(y_true, axis=1, indices=[1])
        day3 = tf.gather(y_true, axis=1, indices=[2])
        day4 = tf.gather(y_true, axis=1, indices=[3])
        day5 = tf.gather(y_true, axis=1, indices=[4])
        day6 = tf.gather(y_true, axis=1, indices=[5])
        day7 = tf.gather(y_true, axis=1, indices=[6])
        day8 = tf.gather(y_true, axis=1, indices=[7])
        day9 = tf.gather(y_true, axis=1, indices=[8])
        day10 = tf.gather(y_true, axis=1, indices=[9])
        day11 = tf.gather(y_true, axis=1, indices=[10])
        day12 = tf.gather(y_true, axis=1, indices=[11])
        day13 = tf.gather(y_true, axis=1, indices=[12])
        day14 = tf.gather(y_true, axis=1, indices=[13])
        day15 = tf.gather(y_true, axis=1, indices=[14])
        day16 = tf.gather(y_true, axis=1, indices=[15])
        day17 = tf.gather(y_true, axis=1, indices=[16])
        day18 = tf.gather(y_true, axis=1, indices=[17])
        day19 = tf.gather(y_true, axis=1, indices=[18])
        day20 = tf.gather(y_true, axis=1, indices=[19])
        day21 = tf.gather(y_true, axis=1, indices=[20])

        day1 = tf.gather(day1, axis=2, indices=[2, 3, 4])
        day2 = tf.gather(day2, axis=2, indices=[2, 3, 4])
        day3 = tf.gather(day3, axis=2, indices=[2, 3, 4])
        day4 = tf.gather(day4, axis=2, indices=[2, 3, 4])
        day5 = tf.gather(day5, axis=2, indices=[2, 3, 4])
        day6 = tf.gather(day6, axis=2, indices=[2, 3, 4])
        day7 = tf.gather(day7, axis=2, indices=[2, 3, 4])
        day8 = tf.gather(day8, axis=2, indices=[2, 3, 4])
        day9 = tf.gather(day9, axis=2, indices=[2, 3, 4])
        day10 = tf.gather(day10, axis=2, indices=[2, 3, 4])
        day11 = tf.gather(day11, axis=2, indices=[2, 3, 4])
        day12 = tf.gather(day12, axis=2, indices=[2, 3, 4])
        day13 = tf.gather(day13, axis=2, indices=[2, 3, 4])
        day14 = tf.gather(day14, axis=2, indices=[2, 3, 4])
        day15 = tf.gather(day15, axis=2, indices=[2, 3, 4])
        day16 = tf.gather(day16, axis=2, indices=[2, 3, 4])
        day17 = tf.gather(day17, axis=2, indices=[2, 3, 4])
        day18 = tf.gather(day18, axis=2, indices=[2, 3, 4])
        day19 = tf.gather(day19, axis=2, indices=[2, 3, 4])
        day20 = tf.gather(day20, axis=2, indices=[2, 3, 4])
        day21 = tf.gather(day21, axis=2, indices=[2, 3, 4])

        day1_pred = tf.gather(y_pred, axis=1, indices=[0])
        day2_pred = tf.gather(y_pred, axis=1, indices=[1])
        day3_pred = tf.gather(y_pred, axis=1, indices=[2])
        day4_pred = tf.gather(y_pred, axis=1, indices=[3])
        day5_pred = tf.gather(y_pred, axis=1, indices=[4])
        day6_pred = tf.gather(y_pred, axis=1, indices=[5])
        day7_pred = tf.gather(y_pred, axis=1, indices=[6])
        day8_pred = tf.gather(y_pred, axis=1, indices=[7])
        day9_pred = tf.gather(y_pred, axis=1, indices=[8])
        day10_pred = tf.gather(y_pred, axis=1, indices=[9])
        day11_pred = tf.gather(y_pred, axis=1, indices=[10])
        day12_pred = tf.gather(y_pred, axis=1, indices=[11])
        day13_pred = tf.gather(y_pred, axis=1, indices=[12])
        day14_pred = tf.gather(y_pred, axis=1, indices=[13])
        day15_pred = tf.gather(y_true, axis=1, indices=[14])
        day16_pred = tf.gather(y_true, axis=1, indices=[15])
        day17_pred = tf.gather(y_true, axis=1, indices=[16])
        day18_pred = tf.gather(y_true, axis=1, indices=[17])
        day19_pred = tf.gather(y_true, axis=1, indices=[18])
        day20_pred = tf.gather(y_true, axis=1, indices=[19])
        day21_pred = tf.gather(y_true, axis=1, indices=[20])

        day1_pred = tf.gather(day1_pred, axis=2, indices=[2, 3, 4])
        day2_pred = tf.gather(day2_pred, axis=2, indices=[2, 3, 4])
        day3_pred = tf.gather(day3_pred, axis=2, indices=[2, 3, 4])
        day4_pred = tf.gather(day4_pred, axis=2, indices=[2, 3, 4])
        day5_pred = tf.gather(day5_pred, axis=2, indices=[2, 3, 4])
        day6_pred = tf.gather(day6_pred, axis=2, indices=[2, 3, 4])
        day7_pred = tf.gather(day7_pred, axis=2, indices=[2, 3, 4])
        day8_pred = tf.gather(day8_pred, axis=2, indices=[2, 3, 4])
        day9_pred = tf.gather(day9_pred, axis=2, indices=[2, 3, 4])
        day10_pred = tf.gather(day10_pred, axis=2, indices=[2, 3, 4])
        day11_pred = tf.gather(day11_pred, axis=2, indices=[2, 3, 4])
        day12_pred = tf.gather(day12_pred, axis=2, indices=[2, 3, 4])
        day13_pred = tf.gather(day13_pred, axis=2, indices=[2, 3, 4])
        day14_pred = tf.gather(day14_pred, axis=2, indices=[2, 3, 4])
        day15_pred = tf.gather(day15_pred, axis=2, indices=[2, 3, 4])
        day16_pred = tf.gather(day16_pred, axis=2, indices=[2, 3, 4])
        day17_pred = tf.gather(day17_pred, axis=2, indices=[2, 3, 4])
        day18_pred = tf.gather(day18_pred, axis=2, indices=[2, 3, 4])
        day19_pred = tf.gather(day19_pred, axis=2, indices=[2, 3, 4])
        day20_pred = tf.gather(day20_pred, axis=2, indices=[2, 3, 4])
        day21_pred = tf.gather(day21_pred, axis=2, indices=[2, 3, 4])

        loss = 1 / 1 * tf.reduce_mean(tf.abs(day1 - day1_pred)) + 1 / 2 * tf.reduce_mean(
            tf.abs(day2 - day2_pred)) + 1 / 3 * tf.reduce_mean(tf.abs(day3 - day3_pred)) + 1 / 4 * tf.reduce_mean(
            tf.abs(day4 - day4_pred)) + 1 / 5 * tf.reduce_mean(tf.abs(day5 - day5_pred)) + 1 / 6 * tf.reduce_mean(
            tf.abs(day6 - day6_pred)) + 1 / 7 * tf.reduce_mean(tf.abs(day7 - day7_pred)) + 1 / 8 * tf.reduce_mean(
            tf.abs(day8 - day8_pred)) + 1 / 9 * tf.reduce_mean(tf.abs(day9 - day9_pred)) + 1 / 10 * tf.reduce_mean(
            tf.abs(day10 - day10_pred)) + 1 / 11 * tf.reduce_mean(tf.abs(day11 - day11_pred)) + 1 / 12 * tf.reduce_mean(
            tf.abs(day12 - day12_pred)) + 1 / 13 * tf.reduce_mean(tf.abs(day13 - day13_pred)) + 1 / 14 * tf.reduce_mean(
            tf.abs(day14 - day14_pred)) + 1 / 15 * tf.reduce_mean(tf.abs(day15 - day15_pred)) + 1 / 16 * tf.reduce_mean(
            tf.abs(day16 - day16_pred)) + 1 / 17 * tf.reduce_mean(tf.abs(day17 - day17_pred)) + 1 / 18 * tf.reduce_mean(
            tf.abs(day18 - day18_pred)) + 1 / 19 * tf.reduce_mean(tf.abs(day19 - day19_pred)) + 1 / 20 * tf.reduce_mean(
            tf.abs(day20 - day20_pred)) + 1 / 21 * tf.reduce_mean(tf.abs(day21 - day21_pred))
        return loss

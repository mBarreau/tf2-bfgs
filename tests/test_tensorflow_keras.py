import tensorflow as tf
import numpy as np
from tf2_bfgs import LBFGS

tf.keras.utils.set_random_seed(1234)


def test_keras():
    t = np.linspace(0, 1, 10).reshape((-1, 1)).astype(np.float32)
    x = np.cos(t)

    # Keras

    omega = tf.keras.Sequential(
        [tf.keras.Input(shape=[1,]),
         tf.keras.layers.Dense(10, "tanh"),
         tf.keras.layers.Dense(10, "tanh"),
         tf.keras.layers.Dense(10, "tanh"),
         tf.keras.layers.Dense(1, None)])

    def get_cost(model, t, x):
        x_hat = model(t)
        return tf.keras.losses.MeanSquaredError()(x_hat, x)

    optimizer_BFGS = LBFGS(get_cost, omega.trainable_variables)
    optimizer_BFGS.minimize(omega, t, x)

    # Plot

    t_test = np.linspace(0, 1, 50).reshape((-1, 1)).astype(np.float32)
    x_test = np.cos(t_test)
    assert np.linalg.norm(omega(t_test) - x_test) <= 0.01

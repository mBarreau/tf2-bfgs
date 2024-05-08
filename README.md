# tf2-bfgs
> Use BFGS optimizer in Tensorflow 2 almost as it was with Tensorflow 1 and `tf.contrib`

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`tf2-bfgs` is a wrapper to train a tf.Module or tf.keras.Model using (L-)BFGS optimizer from Tensorflow Probability. This work uses the code snippet from Pi-Yueh Chuang available [here](https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993).

## Install

You can run the following command in a dedicated virtual environment:
```
pip install tf2-bfgs
```
or the following one to get the latest version
```
pip install git+https://github.com/mBarreau/tf2-bfgs
```

You might also need to install extra dependencies such as
```
pip install tensorflow-probabilities tf_keras
```

## Examples

We will train a regression model to fit a `cos`. Here are the training data.
```
import tensorflow as tf
import numpy as np
from tf2_bfgs import LBFGS

t = np.linspace(0, 1, 10).reshape((-1, 1)).astype(np.float32)
x = np.cos(t)
```

### Using Keras

We define the model first, a multilayer perceptron.
```
omega = tf.keras.Sequential(
    [tf.keras.Input(shape=[1,]),
     tf.keras.layers.Dense(10, "tanh"),
     tf.keras.layers.Dense(10, "tanh"),
     tf.keras.layers.Dense(10, "tanh"),
     tf.keras.layers.Dense(1, None)])
```

The loss function is the traditional mean squared error:
```
def get_cost(model, t, x):
    x_hat = model(t)
    return tf.keras.losses.MeanSquaredError()(x_hat, x)
```

**_NOTE:_** for differentiablility reasons, the output of the model `x_hat = model(t)` should be computed inside the cost function.

```
optimizer_BFGS = LBFGS(get_cost, omega.trainable_variables)
results = optimizer_BFGS.minimize(omega, t, x)
```

The number of parameters to the minimize function is the same as those of `get_cost`.

### Using Tensorflow tf.Module

We will define our own multilayer perceptron as follows:
```
def init(layers):
    Ws, bs = [], []
    for i in range(len(layers) - 1):
        W = xavier_init(size=[layers[i], layers[i + 1]])
        b = tf.zeros([1, layers[i + 1]])
        Ws.append(tf.Variable(W, dtype=tf.float32, name=f"W_{i}"))
        bs.append(tf.Variable(b, dtype=tf.float32, name=f"b_{i}"))
    return Ws, bs


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return np.random.normal(size=[in_dim, out_dim], scale=xavier_stddev)


class NeuralNetwork(tf.Module):
    def __init__(self, hidden_layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = [1] + hidden_layers + [1]
        self.Ws, self.bs = init(layers=self.layers)

    @tf.function
    def __call__(self, input):
        num_layers = len(self.Ws)
        H = tf.cast(input, tf.float32)
        for layer in range(0, num_layers - 1):
            W = self.Ws[layer]
            b = self.bs[layer]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.Ws[-1]
        b = self.bs[-1]
        return tf.add(tf.matmul(H, W), b)

omega = NeuralNetwork([10]*3)
```

**_NOTE:_** It is of main importance that the class inherits from `tf.Module`.

Define the cost function as follows:
```
@tf.function
def get_cost(model, t, x):
    return tf.reduce_mean(tf.square(model(t) - x))
```

**_NOTE:_** for differentiablility reasons, the output of the model `x_hat = model(t)` should be computed inside the cost function. Note that it is also recommended to use the decorator `@tf.function` for performance reasons.

The training is perfomed by the following snippet:
```
optimizer_BFGS = LBFGS(get_cost, omega.trainable_variables)
results = optimizer_BFGS.minimize(omega, t, x)
```

The number of parameters to the minimize function is the same as those of `get_cost`.

### Using physics-informed neural networks

This optimizer might be useful when dealing with physics-informed neural network. To this end, we must change the cost from the previous subsection to include the minimization of the physics residual.

```
@tf.function
def get_pinn_cost(model, t, x, t_phys):
    # Data cost
    data_cost = tf.reduce_mean(tf.square(model(t) - x))
    # Physics cost
    t_phys = tf.convert_to_tensor(t)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(t)
        model_tf = model(t)
    dmodel_dt = tape.gradient(model_tf, t_phys)
    physics_cost = tf.reduce_mean(tf.square(dmodel_dt + tf.sin(t)))
    return data_cost + 0.1 * physics_cost
```

Training is done in a similar manner.
```
t_phys = np.linspace(0, 1, 100).reshape((-1, 1)).astype(np.float32)
optimizer_BFGS = LBFGS(get_pinn_cost, omega.trainable_variables)
results = optimizer_BFGS.minimize(omega, t, x, t_phys)
```

## Options

The optimizer has the same options as the function `tfp.optimizer.bfgs_minimize` in the case of `BFGS` or `tfp.optimizer.lbfgs_minimize` in the case of `LBFGS`.

For instance, to set the maximum number of iterations to 100, we can use the following code when defining the optimizer:
```
options = {'max_iterations': 100}
optimizer_BFGS = LBFGS(get_cost, omega.trainable_variables, options)
```


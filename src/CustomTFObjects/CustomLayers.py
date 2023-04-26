import tensorflow as tf
from typing import Dict, Any, Type, Tuple


@tf.keras.utils.register_keras_serializable()
class DensePerturbated(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        activation: str = "linear",
        random_prior: str = "gaussian",
        magnitude: float = 0,
        trainable: bool = True,
        name: str = None,
        dtype: Type = None,
        dynamic: bool = False,
        **kwargs,
    ):
        super(DensePerturbated, self).__init__(
            trainable, name, dtype, dynamic, **kwargs
        )
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.random_prior = random_prior
        self.magnitude = magnitude

    def build(self, input_shape: Tuple[int]) -> None:
        super().build(input_shape)
        self.kernel = self.add_weight(
            name=self.name + "/kernel",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                name=self.name + "/bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        self.input_last_dim = input_shape[-1]
        self.inputs_shape = tuple([32] + list(input_shape)[1:])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["units"] = self.units
        config["use_bias"] = self.use_bias
        config["activation"] = self.activation
        config["random_prior"] = self.random_prior
        config["magnitude"] = self.magnitude
        return config

    def get_noise(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.random_prior == "gaussian":
            return (
                tf.reduce_max(inputs)
                * self.magnitude
                * tf.random.normal(self.inputs_shape, mean=0, stddev=1)
            )
        elif self.random_prior == "uniform":
            return (
                tf.reduce_max(inputs)
                * self.magnitude
                * tf.random.uniform(self.inputs_shape)
            )
        elif self.random_prior == "dirac":
            return (
                tf.reduce_max(inputs)
                * self.magnitude
                * tf.one_hot(
                    indices=tf.random.uniform(
                        shape=(32,) if len(self.inputs_shape) == 2 else (32, 1),
                        minval=0,
                        maxval=self.input_last_dim,
                        dtype=tf.int32,
                    ),
                    depth=self.input_last_dim,
                )
            )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        noise = self.get_noise(inputs=inputs)
        output = tf.linalg.matmul(a=inputs + noise, b=self.kernel)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)
        if self.activation != "linear":
            if isinstance(self.activation, str):
                output = tf.keras.layers.Activation(self.activation)(output)
            else:
                output = self.activation(output)
        return output


@tf.keras.utils.register_keras_serializable()
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, probability: float = 0.5, **kwargs):
        super(StochasticDepth, self).__init__()
        self.probability = probability
        self.add_layer = tf.keras.layers.Add()

    def get_config(self):
        config = super().get_config()
        config["probability"] = self.probability
        return config

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        coin_toss = tf.random.uniform(())
        return tf.cond(
            tf.greater(self.probability, coin_toss),
            lambda: inputs[0],
            lambda: self.add_layer(inputs),
        )

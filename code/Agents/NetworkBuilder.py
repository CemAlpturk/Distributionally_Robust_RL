import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Dense, Lambda, Input, Add
from tensorflow.keras.models import load_model


# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K


class NetworkBuilder:

    @staticmethod
    def build(network_parameters: dict) -> Sequential:
        """
        Builds a network with specified parameters.
        """
        assert isinstance(network_parameters,
                          dict), "Invalid input type. Expecting argument 'network_parameters' to be of type 'dict'"
        NetworkBuilder._validate_network_parameters(network_parameters)

        optimizers = {
            "adam": Adam(lr=network_parameters["learning_rate"])
        }

        model = Sequential()
        model.add(Input(shape=network_parameters["input_shape"]))
        init = network_parameters["initializer"]
        for layer in network_parameters["layers"]:
            nodes, activation_function = layer
            model.add(Dense(nodes, activation=activation_function, kernel_initializer=init))

        model.compile(
            loss=network_parameters["loss_function"],
            optimizer=optimizers[network_parameters["optimizer"]])

        return model

    @staticmethod
    def build_dueling(network_parameters: dict):
        assert isinstance(network_parameters,
                          dict), "Invalid input type. Expecting argument 'network_parameters' to be of type 'dict'"
        NetworkBuilder._validate_network_parameters(network_parameters)

        optimizers = {
            "adam": Adam(lr=network_parameters["learning_rate"])
        }
        X_Input = Input(shape=network_parameters['input_shape'])
        X = X_Input

        init = network_parameters["initializer"]
        for layer in network_parameters["layers"]:
            nodes, activation = layer
            X = Dense(nodes, activation=activation, kernel_initializer=init)(X)

        # Dueling layers
        V = Dense(1, kernel_initializer=init)(X)
        V = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                             output_shape=(network_parameters["output_size"],))(V)
        A = Dense(network_parameters["output_size"], kernel_initializer=init)(X)
        A = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                   output_shape=(network_parameters["output_size"],))(A)
        X = Add()([V, A])

        model = Model(inputs=X_Input, outputs=X, name='Dueling_DQN')
        model.compile(
            loss=network_parameters["loss_function"],
            optimizer=optimizers[network_parameters["optimizer"]],
            metrics=["accuracy"])

        model.summary()
        return model

    @staticmethod
    def _validate_network_parameters(network_parameters: dict):
        """
        Validates inputs.
        TODO: Complete validation -- or ignore if it's too much.
        """

        assert 'layers' in network_parameters, "Missing key 'layers' in network parameters"
        assert isinstance(network_parameters['layers'], list), "Invalid type, expected 'layers' to be of type 'list'"

        assert 'input_shape' in network_parameters, "Missing key 'input_shape' in network parameters"
        assert 'loss_function' in network_parameters, "Missing key 'loss_function' in network parameters"
        assert 'optimizer' in network_parameters, "Missing key 'optimizer' in network parameters"

    @staticmethod
    def load_model(filepath):
        """
        Loads preexisting model
        """
        return load_model(filepath)


if __name__ == '__main__':
    import numpy as np

    optimizer = Adam()
    network_parameters = {
        "input_shape": (4,),  # Network input shape.
        "output_size": 4,
        "layers": [(10, 'relu'), (10, 'relu')],
        "optimizer": "adam",  # optimizer
        "loss_function": "mse",
        "initializer": tf.keras.initializers.he_uniform()
    }

    model = NetworkBuilder.build_dueling(network_parameters)


    x = np.random.rand(1, 4)
    print(x)
    print(model.predict(x))

import tensorflow as tf
from tensorflow import keras
# noinspection PyUnresolvedReferences
from tensorflow.keras import Sequential
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Dense
# noinspection PyUnresolvedReferences
from tensorflow.keras import Input
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


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
            "adam": Adam()
        }

        model = Sequential()
        model.add(Input(shape=network_parameters["input_shape"]))
        # init = network_parameters["initializer"]
        for layer in network_parameters["layers"]:
            nodes, activation_function = layer
            model.add(Dense(nodes, activation=activation_function))

        model.compile(
            loss=network_parameters["loss_function"],
            optimizer=optimizers[network_parameters["optimizer"]])

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
    def _load_model(filepath):
        """
        Loads preexisting model
        """
        return load_model(filepath)


if __name__ == '__main__':
    from tensorflow.keras.optimizers import Adam
    import numpy as np

    optimizer = Adam()
    network_parameters = {
        "input_shape": (4,),  # Network input shape.
        "layers": [(10, 'relu'), (1, 'linear')],
        "optimizer": optimizer,  # optimizer
        "loss_function": "mse"
    }

    model = NetworkBuilder.build(network_parameters)
    print(model.summary())

    x = np.random.rand(1, 4)
    print(x)
    print(model.predict(x))




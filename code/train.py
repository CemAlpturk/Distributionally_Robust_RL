import numpy as np

from Environment.Environment import Environment
from Agents.DRAgent import DRAgent
import tensorflow as tf
import keras.backend as K

from Utilities import plots

env = Environment()


def custom_loss_function(y_true, y_pred):
    """
    Only the loss from the taken action affects the loss
    """
    # find the nonzero component of y_true
    idx = K.switch(K.not_equal(y_true, 0.0), y_pred, 0.0)
    loss = tf.subtract(y_true, idx)
    return K.square(K.sum(loss))


network_parameters = {
    "input_shape": (6,),
    "layers": [(512, 'relu'), (512, 'relu')],
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss_function": "mse",
    # "loss_function": custom_loss_function,
    "initializer": tf.keras.initializers.he_uniform(),
    "dueling": True,
    "output_size": 4
}

agent = DRAgent(network_parameters, env)
# agent.set_state_lims(env.get_state_lims())
agent.train(
    max_episodes=10000,
    exploration_rate=1.0,
    exploration_rate_decay=0.999,
    min_exploration_rate=0.1,
    stochastic=True,
    discount=0.9,
    batch_size=32,
    max_time_steps=50,
    warm_start=False,
    best=False,
    timedir='2022-03-10_20-40-56',
    model_allignment_period=100,
    evaluate_model_period=500,
    evaluation_size=20,
    lamb=5,
    d_lamb=0.01,
    max_lamb=20,
    render=False,
    save_animation_period=1000
)

# path = agent.Logger.env_param_dir
# plots.plot_vector_field(path, env, agent)

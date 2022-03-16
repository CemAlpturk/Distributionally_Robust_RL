import numpy as np

from Environment.Environment import Environment
from Agents.DRAgent import DRAgent
import tensorflow as tf
import keras.backend as K

from Utilities import plots

# Noise dist
cov = 0.1*np.identity(2)
num_actions = 8
obstacles = [([-5, -5], 2), ([5, -5], 2), ([5, 5], 2), ([-5, 5],2)]
lims = [[-20, 20], [-20, 20]]
env = Environment(num_actions=num_actions, cov=cov, obstacles=obstacles, lims=lims)
num_states = env.state_size


def custom_loss_function(y_true, y_pred):
    """
    Only the loss from the taken action affects the loss
    """
    # find the nonzero component of y_true
    idx = K.switch(K.not_equal(y_true, 0.0), y_pred, 0.0)
    loss = tf.subtract(y_true, idx)
    return K.square(K.sum(loss))


network_parameters = {
    "input_shape": (num_states,),
    "layers": [(512, 'relu'), (512, 'relu')],
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "loss_function": "mse",
    # "loss_function": custom_loss_function,
    "initializer": tf.keras.initializers.he_uniform(),
    "dueling": True,
    "output_size": num_actions
}

agent = DRAgent(network_parameters, env)
# agent.set_state_lims(env.get_state_lims())
agent.train(
    max_episodes=20000,
    exploration_rate=1.0,
    exploration_rate_decay=0.9995,
    min_exploration_rate=0.1,
    stochastic=False,
    discount=0.9,
    batch_size=32,
    max_time_steps=40,
    warm_start=False,
    best=True,
    timedir='2022-03-14_16-45-49',
    model_allignment_period=50,
    evaluate_model_period=250,
    evaluation_size=50,
    lamb=5,
    d_lamb=0.01,
    max_lamb=20,
    render=False,
    save_animation_period=1000
)

# path = agent.Logger.env_param_dir
# plots.plot_vector_field(path, env, agent)

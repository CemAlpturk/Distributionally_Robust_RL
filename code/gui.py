import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import torch
from torch import nn, Tensor

from Agents.DQNLightning import DQNLightning
from Environments.Environment import Environment
from Utilities.plots import plot_multiple_initial_positions, \
    plot_values, plot_from_network, plot_values_from_network


# Import for DRDQN models, not needed if matlab engine is installed
class DQN(nn.Module):
    """
    Dense DQN Network
    """

    def __init__(self,
                 state_size: int,
                 n_actions: int,
                 hidden_size: int = 100,
                 dueling: bool = False,
                 dueling_max: bool = True,
                 init_scale: float = 1.0):
        """
        :param state_size: Input size
        :param n_actions: Output size
        :param hidden_size: Number of hidden neurons
        :param dueling: Dueling architecture
        :param dueling_max: Whether to use max or mean in aggregation layer
        :param init_scale: Initial parameter scaling
        """
        super(DQN, self).__init__()
        self.dueling = dueling
        self.dueling_max = dueling_max
        self.init_scale = init_scale
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.relu = nn.ReLU()

        if dueling:
            self.value = nn.Linear(hidden_size, 1)
            self.adv = nn.Linear(hidden_size, n_actions)
        else:
            self.fc3 = nn.Linear(hidden_size, n_actions)

        # Weight Initialization
        self.apply(self.initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass
        :param x: Input tensor
        :return: Output tensor
        """
        x = self.fc1(x.float())
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        if self.dueling:
            value = self.value(x)
            adv = self.adv(x)

            if self.dueling_max:
                agg, _ = torch.max(adv, dim=1, keepdim=True)
            else:
                agg = torch.mean(adv, dim=1, keepdim=True)
            Q = value + adv - agg
        else:
            Q = self.fc3(x)

        return Q

    def initialize_weights(self, m):
        """
        HE initialization
        :param m: weights
        """
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            n = m.in_features
            y = self.init_scale / np.sqrt(n)
            nn.init.uniform_(m.weight, -y, y)
            # m.weight.data.uniform_(-y,y)
            # m.bias.data.fill_(0.0)


class GUI:
    def __init__(self):
        self.root_x = 800
        self.root_y = 600

        self.image_x = 300
        self.image_y = 300

        self.image_border = 5
        self.image_anchor_x = 0
        self.image_anchor_y = 0

        self.model = None
        self.env = None
        self.x = 0
        self.y = 0
        self.selected_model = False

        # Root
        self.root = Tk()
        # self.root.geometry(f"{self.root_x}x{self.root_y}")
        self.root.title('Demo')
        self.root.resizable(False, False)

        self.plot_frame = Frame(self.root)
        self.plot_frame.grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
        self.root.bind("<Button 1>", self.getorigin)
        fig = self.get_empty_plot()
        self.chart = FigureCanvasTkAgg(fig, self.plot_frame)
        self.chart.get_tk_widget().grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
        # self.chart.show()
        plt.close(fig)

        # Simulate Button
        self.button_simulate = Button(self.root, text="Simulate", command=self.simulate)
        # self.button_simulate.grid(column=3, row=0, sticky=tk.E, padx=5, pady=5)
        # self.button_simulate.pack()

        # Randomize Button
        self.button_randomize = Button(self.root, text="Randomize")
        # self.button_randomize.pack()

        # Initial Position Button ?
        self.button_initial = Button(self.root, text="Initial")
        # self.button_initial.pack()

        # Reset Button
        self.button_reset = Button(self.root, text="Reset")
        # self.button_reset.pack()

        self.button_q = Button(self.root, text="Q_values", command=self.get_qvals)

        # Exit Button
        self.button_exit = Button(self.root, text="Exit", command=self.root.destroy)
        # self.button_exit.pack()

        # Save Button
        self.button_save = Button(self.root, text="Save")
        # self.button_save.pack()

        # Heatmap Checkbox
        self.heatmap = IntVar()
        self.box_heatmap = Checkbutton(self.root,
                                       text="Heatmap",
                                       offvalue=False,
                                       onvalue=True,
                                       variable=self.heatmap)
        # self.box_heatmap.pack()

        # Vector Field Checkbox
        self.vector = IntVar()
        self.box_vector = Checkbutton(self.root,
                                      text="Vector Field",
                                      offvalue=False,
                                      onvalue=True,
                                      variable=self.vector)
        # self.box_vector.pack()

        # Model Selection
        model_list = self.get_models()
        # self.current_model = StringVar()
        self.combo_model = Combobox(self.root,
                                    values=model_list)
        # textvariable=self.current_model)
        self.combo_model['state'] = 'readonly'
        self.combo_model.bind('<<ComboboxSelected>>', self.load_model)
        # self.button_model = Button(self.root, text="Load Model")

        self.scale = Scale(self.root, from_=0.0, to=0.2, orient=HORIZONTAL, command=self.get_scale)
        self.scale_label = Label(self.root, text="Cov = 0.00")

        # Radio Buttons
        self.model_type = IntVar()
        self.radio_DQN = Radiobutton(self.root,
                                     text="DQN",
                                     value=0,
                                     variable=self.model_type,
                                     command=self.get_model_list)
        self.radio_DRDQN = Radiobutton(self.root,
                                       text="DRDQN",
                                       value=1,
                                       variable=self.model_type,
                                       command=self.get_model_list)

        self.place()
        self.root.mainloop()

    def place(self):
        # self.plot.grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
        self.combo_model.grid(column=1, row=1, padx=5, pady=5,
                              rowspan=2)
        # self.button_model.grid(column=1, row=2, sticky=E + W,  padx=5, pady=5)
        self.button_simulate.grid(column=1, row=3, padx=5, pady=5)
        self.button_randomize.grid(column=2, row=3, padx=5, pady=5)
        self.button_q.grid(column=1, row=4, padx=5, pady=5)
        self.button_initial.grid(column=1, row=5, padx=5, pady=5)
        self.box_heatmap.grid(column=1, row=6, padx=5, pady=5)
        self.box_vector.grid(column=1, row=7, padx=5, pady=5)
        self.scale.grid(column=1, row=8, padx=5, pady=5)
        self.scale_label.grid(column=1, row=9, padx=5, pady=5)
        self.button_save.grid(column=1, row=10, padx=5, pady=5)
        self.button_exit.grid(column=2, row=10, padx=5, pady=5)
        self.radio_DQN.grid(column=1, row=0, padx=5, pady=5)
        self.radio_DRDQN.grid(column=2, row=0, padx=5, pady=5)

    @staticmethod
    def get_models():
        models_path = 'models/DQN'
        models = os.listdir(models_path)
        return models

    @staticmethod
    def get_empty_plot():
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', xlim=[-10, 10], ylim=[-10, 10])
        ax.grid()
        ax.set_axisbelow(True)
        return fig

    def get_model_list(self):
        if self.model_type.get() == 0:
            models = self.get_models()
        else:
            models_path = 'models/DRDQN'
            models = os.listdir(models_path)
        self.combo_model['values'] = models

    def load_model(self, *args):
        model_name = self.combo_model.get()
        if model_name == "":
            print('No model selected')
            return
        try:
            # DQN
            if self.model_type.get() == 0:
                model_path = f"models/DQN/{model_name}/checkpoints/last.ckpt"
                self.model = DQNLightning.load_from_checkpoint(model_path)

            # DRDQN
            else:
                model_path = f"models/DRDQN/{model_name}"
                self.model = DQN(state_size=8, n_actions=9, hidden_size=150)
                self.model.load_state_dict(torch.load(model_path))

        except Exception as e:
            messagebox.showerror("Model Error", "An error ocured while loading model")
            print(e)
            return

        self.selected_model = True
        self.simulate()

    def simulate(self, *args):
        if not self.selected_model:
            return
        # Create Environment
        cov = self.scale.get() * np.identity(2)
        num_actions = 8
        obstacles = [(np.array([-3.5, 0.0]), 2), (np.array([3.5, 0.0]), 2)]
        goal = [0.0, 5.0]
        lims = [[-10, 10], [-10, 10]]
        env = Environment(num_actions=num_actions,
                          cov=cov,
                          lims=lims,
                          obstacles=obstacles,
                          static_obs=True,
                          goal=goal)
        self.env = env
        if self.model_type.get() == 0:
            self.model.agent.reset()
            p = [self.x, self.y]
            # p = [0.0, -5.0]
            s = env.state
            s[0:2] = np.array(p)
            env.state = s
            env.robot._x = np.array(p).reshape(-1, 1)
            self.model.env = env
            self.model.agent.env = env
            self.model.agent.state = env.state

            states = [self.model.agent.state]
            net = self.model.net

            n_steps = 50
            episode_reward = 0.0
            for step in range(n_steps):
                reward, done, goal = self.model.agent.play_step(net, epsilon=0.0)
                episode_reward += reward
                states.append(self.model.agent.state)
                if done:
                    break
            trajectory = [np.array(states)]
            fig = plot_multiple_initial_positions(env,
                                                  self.model,
                                                  trajectory,
                                                  vector_field=self.vector.get(),
                                                  heatmap=self.heatmap.get())

            self.chart = FigureCanvasTkAgg(fig, self.plot_frame)
            self.chart.get_tk_widget().grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
            # self.chart.get_tk_widget().pack()
            plt.close(fig)

        else:
            # DRDQN simulation
            p = [self.x, self.y]
            s = self.env.state
            s[0:2] = np.array(p)
            env.state = s
            env.robot._x = np.array(p).reshape(-1, 1)
            states = [s]

            n_steps = 50
            episode_reward = 0.0

            for step in range(n_steps):
                # Select action with null-policy
                state = torch.tensor(np.array([s])).to('cpu')
                q_values = self.model(state)
                _, action = torch.max(q_values[:, :-1], dim=1)
                action = int(action.item())
                new_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward

                s = new_state
                states.append(s)
                if done:
                    break

            trajectory = [np.array(states)]
            fig = plot_from_network(self.model,
                                    env,
                                    trajectory,
                                    vector_field=self.vector.get(),
                                    heatmap=self.heatmap.get())

            self.chart = FigureCanvasTkAgg(fig, self.plot_frame)
            self.chart.get_tk_widget().grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
            # self.chart.get_tk_widget().pack()
            plt.close(fig)

    def getorigin(self, eventorigin):
        if self.env is None:
            return
        x = eventorigin.x
        y = -eventorigin.y
        print(f"Event: ({x},{y})")
        x_lim = [108, 385]
        y_lim = [-322, -45]
        x_grad = (self.env.x_max - self.env.x_min) / (x_lim[1] - x_lim[0])
        y_grad = (self.env.y_max - self.env.y_min) / (y_lim[1] - y_lim[0])
        plot_x = x * x_grad - 108 - self.env.x_min + 80.15
        plot_y = y * y_grad - 45 - self.env.y_min + 48.1

        # Check
        if self.env.x_min <= plot_x <= self.env.x_max and self.env.y_min <= plot_y <= self.env.y_max:
            self.x = plot_x
            self.y = plot_y
            self.simulate()

    def get_qvals(self, *args):
        if not self.selected_model:
            return

        if self.model_type.get() == 0:

            fig = plot_values(self.env, self.model, show_env=True)
            self.chart = FigureCanvasTkAgg(fig, self.plot_frame)
            self.chart.get_tk_widget().grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)

            plt.close(fig)

        else:
            fig = plot_values_from_network(self.model, self.env, show_env=True)
            self.chart = FigureCanvasTkAgg(fig, self.plot_frame)
            self.chart.get_tk_widget().grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)

            plt.close(fig)

    def get_scale(self, *args):
        selection = f"Cov = {self.scale.get():0.2f}"
        self.scale_label.config(text=selection)


if __name__ == '__main__':
    gui = GUI()

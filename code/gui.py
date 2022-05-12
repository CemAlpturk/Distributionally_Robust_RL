import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk

from Agents.DQNLightning import DQNLightning
from Environments.Environment import Environment
from Utilities.plots import plot_multiple_initial_positions


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

        # Root
        self.root = Tk()
        # self.root.geometry(f"{self.root_x}x{self.root_y}")
        self.root.title('Demo')
        self.root.resizable(False, False)


        fig = self.get_empty_plot()
        self.chart = FigureCanvasTkAgg(fig, self.root)
        self.chart.get_tk_widget().grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
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

        # Exit Button
        self.button_exit = Button(self.root, text="Exit")
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

        self.place()
        self.root.mainloop()

    def place(self):
        # self.plot.grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
        self.combo_model.grid(column=1, row=0, sticky=E + W, padx=5, pady=5,
                              rowspan=2)
        # self.button_model.grid(column=1, row=2, sticky=E + W,  padx=5, pady=5)
        self.button_simulate.grid(column=1, row=2, sticky=E + W, padx=5, pady=5)
        self.button_randomize.grid(column=1, row=3, sticky=E + W, padx=5, pady=5)
        self.button_reset.grid(column=1, row=4, sticky=E + W, padx=5, pady=5)
        self.button_initial.grid(column=1, row=5, sticky=E + W, padx=5, pady=5)
        self.box_heatmap.grid(column=1, row=6, sticky=E + W, padx=5, pady=5)
        self.box_vector.grid(column=1, row=7, sticky=E + W, padx=5, pady=5)
        self.button_save.grid(column=1, row=8, sticky=E + W, padx=5, pady=5)
        self.button_exit.grid(column=1, row=9, sticky=E + W, padx=5, pady=5)

    @staticmethod
    def get_models():
        models_path = 'lightning_logs'
        models = os.listdir(models_path)
        return models

    @staticmethod
    def get_empty_plot():
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', xlim=[-10, 10], ylim=[-10, 10])
        ax.grid()
        ax.set_axisbelow(True)
        return fig

    def load_model(self, *args):
        model_name = self.combo_model.get()
        if model_name == "":
            print('No model selected')
            return
        print(model_name)
        model_path = f"lightning_logs/{model_name}/checkpoints/last.ckpt"
        try:
            self.model = DQNLightning.load_from_checkpoint(model_path)
        except:
            messagebox.showerror("Model Error", "An error ocured while loading model")
            return

        self.simulate()

    def simulate(self, *args):
        # Create Environment
        cov = 0.15 * np.identity(2)
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
        self.model.agent.reset()
        p = [0.0, -5.0]
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
        fig = plot_multiple_initial_positions(env, self.model, trajectory)

        self.chart = FigureCanvasTkAgg(fig, self.root)
        self.chart.get_tk_widget().grid(column=0, row=0, sticky=E + W, columnspan=1, rowspan=10)
        plt.close(fig)

if __name__ == '__main__':
    gui = GUI()

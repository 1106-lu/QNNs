from qnn.embedding.optimization import cost, g_finitedifference, g_parametershift
from qnn.embedding.circuits import sample
from typing import List, Optional, Union, Any
from qnn.qutip_extras.plot_bloch import add_binary_points
from numpy import ndarray

import cirq
import qutip as qt
import matplotlib.pyplot as plt


class Training:

    def __init__(self,
                 circuits: List[cirq.Circuit],
                 learning_rate: float,
                 epsilon: float,
                 epoch: int,
                 initial_parameters: List[Union[float, Any]] or Union[ndarray, int, float, complex] or None,
                 plot_bloch: Optional[bool] = True,
                 parameter_shift: Optional[bool] = False):
        self.circuits = circuits
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epoch = epoch
        self.initial_parameters = initial_parameters
        self.plot_bloch = plot_bloch
        self.parameter_shift = parameter_shift

    @property
    def train(self):
        sample_vectors = sample(cs=self.circuits, theta=self.initial_parameters)

        if self.plot_bloch:
            a = sample_vectors[10:]
            b = sample_vectors[:10]
            be = qt.Bloch()
            be = add_binary_points(a, b, be)
            if isinstance(be, qt.Bloch):
                be.render(be.fig, be.axes)
                plt.show()
                be.save(name='/Users/usuario/Desktop/QIT/QNNs/qnn/embedding/data_tmp/before.png', format='png')



        self.cost_plot, self.epoch_plot, self.params_plot = [], [], []
        #bar_func = progressbar.ProgressBar(maxval=self.epoch)

        params = self.initial_parameters
        lr = self.learning_rate
        for o in range(self.epoch):
            for i in range(len(params)):
                if self.parameter_shift:
                    params[i] = lr*params[i] + (1-lr)*(g_parametershift(i, self.circuits, params))**2
                else:
                    params[i] = params[i].real - lr * g_finitedifference(i, self.circuits, params, self.epsilon).real

            sample_vectors = sample(self.circuits, params)
            self.cost_plot.append(cost(sample_vectors))
            self.epoch_plot.append(o)
            self.params_plot.append(params)

            if o != 0:
                if self.cost_plot[o] < self.cost_plot[o - 1]:
                    lr = lr + lr*2
                else:
                    lr = lr/2

            print('Epoch:', o, 'Cost:', self.cost_plot[o], 'LearningRate:', lr)

        plt.plot(self.epoch_plot, self.cost_plot)
        plt.show()
        return

    def minima(self):
        min_i = 0
        min_cost = self.cost_plot[0]
        for i in range(len(self.cost_plot)):
            if self.cost_plot[i] < min_cost:
                min_cost = self.cost_plot[i]
                min_i = i

        print('min COST:', min_cost)
        print('min EPOCH:', min_i)
        print('min PARAM:', self.params_plot[min_i])
        
        sample_vectors_final = sample(self.circuits, self.params_plot[min_i])
        a_final = sample_vectors_final[int((len(sample_vectors_final)) / 2):]
        b_final = sample_vectors_final[:int((len(sample_vectors_final)) / 2)]
        af = qt.Bloch()
        af = add_binary_points(a_final, b_final, af)
        af.render(af.fig, af.axes)
        plt.show()
        af.save(name='/Users/usuario/Desktop/QIT/QNNs/qnn/embedding/data_tmp/after.png', format='png')
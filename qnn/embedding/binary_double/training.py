from qnn.embedding.optimization import cost
from qnn.embedding.binary_double.circuits import Circuits
from qnn.qutip_extras.plot_bloch import add_binary_points
from typing import List, Optional, Union, Any
from numpy import ndarray
import matplotlib.pyplot as plt
import qutip as qt


class Training:

	def __init__(self,
	             data,
	             depth: int,
	             learning_rate: float,
	             epsilon: float,
	             epoch: int,
	             initial_parameters: List[Union[float, Any]] or Union[ndarray, int, float, complex] or None,
	             plot_bloch: Optional[bool] = True,
	             parameter_shift: Optional[bool] = False):
		self.data = data
		self.depth = depth
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.epoch = epoch
		self.initial_parameters = initial_parameters
		self.plot_bloch = plot_bloch
		self.parameter_shift = parameter_shift
		self.cost_plot, self.epoch_plot, self.params_plot = [], [], []

	@property
	def train(self):
		cs_func = Circuits(data_x=self.data, depth=self.depth, theta=self.initial_parameters, epsilon=self.epsilon)
		cs_func.binary_double()

		sample_vectors = cs_func.sample()
		if self.plot_bloch:
			a = sample_vectors[10:]
			b = sample_vectors[:10]
			be = qt.Bloch()
			be = add_binary_points(a, b, be)
			if isinstance(be, qt.Bloch):
				be.render(be.fig, be.axes)
				plt.show()
				be.save(name='/Users/usuario/Desktop/QIT/QNNs/qnn/embedding/data_tmp/before.png', format='png')

		lr = self.learning_rate
		for o in range(self.epoch):
			for i in range(len(cs_func.theta)):
				if self.parameter_shift:
					cs_func.theta[i] = lr * cs_func.theta[i] + (1 - lr) * (
						cs_func.g_parameter_shift(i, cs_func.theta))**2
				else:
					cs_func.theta[i] = cs_func.theta[i].real - lr * cs_func.g_finite_difference(i, cs_func.theta).real

			sample_vectors = cs_func.sample()
			self.cost_plot.append(cost(sample_vectors))
			self.epoch_plot.append(o)
			self.params_plot.append(cs_func.theta)

			if o != 0:
				if lr < 364500:
					if self.cost_plot[o] < self.cost_plot[o - 1]:
						lr = lr + lr * 2
					else:
						lr = lr / 2
				else:
					lr = lr

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

		cs_func = Circuits(data_x=self.data, depth=self.depth, theta=self.params_plot[min_i], epsilon=self.epsilon)
		cs_func.binary_double()
		sample_vectors_final = cs_func.sample()

		a_final = sample_vectors_final[int((len(sample_vectors_final)) / 2):]
		b_final = sample_vectors_final[:int((len(sample_vectors_final)) / 2)]
		af = qt.Bloch()
		af = add_binary_points(a_final, b_final, af)
		af.render(af.fig, af.axes)
		plt.show()
		af.save(name='/Users/usuario/Desktop/QIT/QNNs/qnn/embedding/data_tmp/after.png', format='png')

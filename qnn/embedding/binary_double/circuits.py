import cirq
import numpy as np
import sympy
from numpy import ndarray
from typing import List, Union, Any, Optional
from qnn.embedding.optimization import cost


class Circuits:
	def __init__(self,
	             data_x: List[Union[float, Any]] or Union[ndarray, int, float, complex],
	             depth: int,
	             parameter_shift: Optional[bool] = False,
	             theta: Optional[List[Union[float, Any]] or Union[ndarray, int, float, complex]] = None,
	             epsilon: Optional[float] = None):

		self.data_x = data_x
		self.depth = depth
		self.theta = theta
		self.cs = None
		self.parameter_shift = parameter_shift
		self.epsilon = epsilon

	def binary_double(self):
		q, self.cs = [], []

		theta_raw = sympy.symbols("theta:200")
		i = 0

		for a in range(len(self.data_x)):
			c = cirq.Circuit()
			for e in range(self.depth):
				c.append(cirq.rx(self.data_x[a])(q[0]))
				c.append(cirq.ry(theta_raw[i])
				         (q[0]))
				i = i + 1
			c.append(cirq.rx(self.data_x[a])(q[0]))
			self.cs.append(c)

		return self.cs

	def g_parameter_shift(self, param, theta_sample):
		perturbation_vector = np.zeros(len(theta_sample))
		perturbation_vector[param] = 1

		neg_theta = theta_sample - perturbation_vector
		pos_theta = theta_sample + perturbation_vector
		neg_vec = self.sample(neg_theta)
		pos_vec = self.sample(pos_theta)

		result = cost(pos_vec) - cost(neg_vec)
		return result

	def g_finite_difference(self, param, theta_sample):
		if self.epsilon is None:
			raise (ValueError('Define epsilon the calculation of the gradient'))
		perturbation_vector = np.zeros(len(theta_sample))
		perturbation_vector[param] = 1

		neg_theta = theta_sample - self.epsilon * perturbation_vector
		pos_theta = theta_sample + self.epsilon * perturbation_vector
		neg_vec = self.sample(theta_sample=neg_theta)
		pos_vec = self.sample(theta_sample=pos_theta)

		result = (cost(pos_vec) - cost(neg_vec)) / 2 * self.epsilon
		return result

	def sample(self,
	           theta_sample: Optional[List[Union[float, Any]] or Union[ndarray, int, float, complex]] = None):
		simulator = cirq.Simulator()
		sample_vectors = []
		resolvers = []
		k = 0

		if self.cs is None:
			raise (ValueError('Create the circuits'))

		if theta_sample is None:
			theta_sample = self.theta
			if self.theta is None:
				raise (ValueError('Define theta for sampling of the circuits'))

		for a in range(len(self.cs)):
			rrange = np.arange(k, k + self.depth, 1)
			resolvers.append(cirq.ParamResolver({'theta' + str(e): theta_sample[e] for e in rrange}))
			k = k + self.depth
			sample_vectors.append(simulator.simulate(program=self.cs[a], param_resolver=resolvers[a]).final_state)

		return sample_vectors


def print_circuits(circuits):
	for i in circuits:
		print(i)

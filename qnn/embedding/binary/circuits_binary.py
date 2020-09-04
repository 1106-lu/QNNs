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

	def binary(self):
		q = []
		for i in range(len(self.data_x)):
			q.append(cirq.GridQubit(i, 0))
		theta_raw = sympy.symbols("theta:500")
		i = 0

		data_a = self.data_x[10:]
		data_b = self.data_x[:10]
		data_len = len(data_a)

		self.cs = cirq.Circuit()
		for _ in range(self.depth):
			for a in range(data_len):
				self.cs.append(cirq.rx(data_a[a])(q[a]))
				self.cs.append(cirq.rx(data_b[a])(q[a+data_len]))
				self.cs.append(cirq.ry(theta_raw[i + 1])(q[a]))
				self.cs.append(cirq.ry(theta_raw[i + 2])(q[a+data_len]))
				i = i + 2

		for a in range(data_len):
			self.cs.append(cirq.rx(data_a[a])(q[a]))
			self.cs.append(cirq.rx(data_b[a])(q[a+data_len]))

		return self.cs

	def g_parameter_shift(self, param, theta_sample):
		perturbation_vector = np.zeros(len(theta_sample))
		perturbation_vector[param] = 1

		neg_theta = theta_sample - perturbation_vector
		pos_theta = theta_sample + perturbation_vector
		neg_vec = sample(neg_theta, circuits=self.cs)
		pos_vec = sample(pos_theta, circuits=self.cs)

		result = cost(pos_vec) - cost(neg_vec)
		return result

	def g_finite_difference(self, param, theta_sample):
		if self.epsilon is None:
			raise (ValueError('Define epsilon the calculation of the gradient'))
		perturbation_vector = np.zeros(len(theta_sample))
		perturbation_vector[param] = 1

		neg_theta = theta_sample - self.epsilon * perturbation_vector
		pos_theta = theta_sample + self.epsilon * perturbation_vector
		neg_vec = sample(theta_sample=neg_theta, circuits=self.cs)
		pos_vec = sample(theta_sample=pos_theta, circuits=self.cs)

		result = (cost(pos_vec) - cost(neg_vec)) / 2 * self.epsilon
		return result

def sample(theta_sample, circuits):

	simulator = cirq.Simulator()
	sample_vectors = []
	resolvers = []
	k, i = 0, 0

	for a in circuits:
		rrange = np.arange(k, k + 12, 1)
		resolvers.append(cirq.ParamResolver({'theta' + str(e): theta_sample[e] for e in rrange}))
		k = k + 12
		sample_vectors.append(simulator.simulate(program=a, param_resolver=resolvers[i]).final_state)
		i = i + 1
	return sample_vectors

def print_circuits(circuits):
	for i in circuits:
		print("__________")
		print(i)

def print_vectors(vectors):
	for i in vectors:
		print("__________")
		print(i)

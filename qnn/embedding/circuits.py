import cirq
import numpy as np
import sympy


class Circuits:
	def __init__(self, depth):
		self.depth = depth

	def create(self, data_x):
		cs = []
		q = []
		for i in range(len(data_x)): q.append(cirq.GridQubit(i, 0))
		theta_raw = sympy.symbols("theta:80")

		i = 0
		for a in range(len(data_x)):
			c = cirq.Circuit()
			for e in range(self.depth):
				c.append(cirq.rx(data_x[a])(q[0]))
				c.append(cirq.ry(theta_raw[i])(q[0]))
				i = i + 1
			c.append(cirq.rx(data_x[a])(q[0]))
			cs.append(c)
		return cs

	def sample(self, cs, theta):
		simulator = cirq.Simulator()
		sample_vectors = []
		resolvers = []
		k = 0
		for a in range(len(cs)):
			rrange = np.arange(k, k + 4, 1)
			resolvers.append(cirq.ParamResolver({'theta' + str(e): theta[e] for e in rrange}))
			k = k + 4
			sample_vectors.append(simulator.simulate(program=cs[a], param_resolver=resolvers[a]).final_state)
		return sample_vectors

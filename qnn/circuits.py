import cirq
import numpy as np
import sympy

class Circuits:
	def __init__(self,
	             data_x):
		self.data_x = data_x

	def create(self, data_x, depth):
		cs = []
		q = []
		for i in range(len(data_x)): q.append(cirq.GridQubit(i, 0))
		theta = sympy.symbols("theta:80")

		i = 0
		for a in range(len(data_x)):
			c = cirq.Circuit()
			for e in range(depth):
				c.append(cirq.rx(data_x[a])(q[0]))
				c.append(cirq.ry(theta[i])(q[0]))
				i = i + 1
			c.append(cirq.rx(data_x[a])(q[0]))
			cs.append(c)

		return cs

def sample(cs, theta):
	simulator = cirq.Simulator()
	sample_vectors = []
	resolvers = []
	i = 0
	for a in range(len(cs)):
		rrange = np.arange(i, i + 4, 1)
		resolvers.append(cirq.ParamResolver({'theta' + str(e): theta[e] for e in rrange}))
		i = i + 4
		sample_vectors.append(simulator.simulate(program=cs[a], param_resolver=resolvers[a]).final_state)
	return sample_vectors

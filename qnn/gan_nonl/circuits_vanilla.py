##
import cirq
import numpy as np
import sympy
from cirq.sim.density_matrix_utils import _probs

from qnn.gan_nonl.ops import Partial_Trace

N = 4
N_a = 2
L = 1

data = [0.44, 0, .55, 0]
q = []

theta = sympy.symbols("theta:1000")

for i in range(4):
	q.append(cirq.GridQubit(i, 0))

c = cirq.Circuit()

for j in [2, 3]:
	c.append(cirq.ry(data[j])(q[j]))

for j in range(2):
	c.append(cirq.I(q[j]))

for _ in range(L):
	for j in range(3):
		c.append(cirq.CZ(q[j + 1], q[j]))

	o = 0
	for j in range(4):
		c.append(cirq.rx(theta[j + o])(q[j]))
		c.append(cirq.ry(theta[j + 1 + o])(q[j]))
		c.append(cirq.rz(theta[j + 2 + o])(q[j]))
		o += 3

##
theta_sample = np.random.normal(0, 2 * np.pi, 15)

resolver = cirq.ParamResolver({'theta' + str(e): theta_sample[e] for e in range(len(theta_sample))})
state_vector = cirq.sim.final_state_vector(program=c, param_resolver=resolver)

##
pt = Partial_Trace(state_vector, 2)
post_pt = pt.compute_matrix()

print(post_pt)

##
probs = []
for l in range(2):
	print(_probs(post_pt, [l], (4, 1)))

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras import layers
from typing import List


def discriminator_model(specs: List):
	model = tf.keras.Sequential()
	model.add(layers.Softmax())

	for n in specs:
		model.add(layers.Dense(n, activation='relu'))

	model.add(layers.Dense(1, activation='relu'))
	model.add(layers.Softmax())

	return model


L = 2
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

c_tensor = tfq.convert_to_tensor([c])

print(tfq.from_tensor(c_tensor))

quantum_model = tfq.layers.PQC(c_tensor)

qc_model = tf.keras.Model()

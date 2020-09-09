import cirq
import matplotlib.pyplot as plt
import numpy as np
import sympy

from qnn.qnlp.optimization import cost_2


def sample_run(circuits: cirq.Circuit, theta_sample: np.array, repetitions):
	a = 1
	if not circuits.has_measurements():
		for u in circuits.all_qubits():
			circuits.append(cirq.measure(u))
			a = a + 1

	resolver = cirq.ParamResolver({'theta1': theta_sample[0]})
	return cirq.Simulator().run(program=circuits, param_resolver=resolver, repetitions=repetitions)


def g_finite_difference(circuits, param, theta_sample: np.array, epsilon: float, expected_bits):
	perturbation_vector = np.zeros(len(theta_sample))
	perturbation_vector[param] = 1

	pos_theta = theta_sample + epsilon * perturbation_vector
	neg_theta = theta_sample - epsilon * perturbation_vector

	pos_result = sample_run(circuits, pos_theta, 1000)
	neg_result = sample_run(circuits, neg_theta, 1000)

	result = (cost_2(pos_result, expected_bits) - cost_2(neg_result, expected_bits)) / 2 * epsilon

	return result


epsilon = .001
cost_plot = []
epoch_plot = []
learning_rate = [1]
expected_bits = [1]

theta1 = sympy.symbols('theta1')
q = cirq.GridQubit(1, 0)

c = cirq.Circuit()
c.append(cirq.rx(theta1)(q))
c.append(cirq.rx(theta1)(q))
c.append(cirq.rx(theta1)(q))
c.append(cirq.rx(theta1)(q))
print(c)

epoch = 110
lr = 100000
parameters = np.random.normal(0, 2 * np.pi, 1)


for o in range(epoch):
	result = sample_run(c, parameters, 1000)
	cost_plot.append(cost_2(result, expected_bits))
	#  cost_plot.append(cost_1(result, 1))
	epoch_plot.append(o)

	for i in range(len(parameters)):
		parameters[i] = parameters[i] + lr * g_finite_difference(c, i, parameters, epsilon, expected_bits)

	#if o != 0:
	#	if (cost_plot[o]) < (cost_plot[o - 1]):
	#		lr = lr * 1.1
	#	else:
	#		lr = lr/2

	print('Epoch:', o, 'Cost:', cost_plot[o], 'LearningRate:', lr)

result = sample_run(c, parameters, 1000)
print(result)
plt.plot(epoch_plot, cost_plot)
plt.show()
print('############################')

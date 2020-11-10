import matplotlib.pyplot as plt
import numpy as np

from qnn.qnlp.circuits_words import CircuitsWords
from qnn.qnlp.optimization import cost_global
from qnn.qnlp.optimization_words import g_parameter_shift_global_words, get_overall_run_words, get_expected_bits

cost_plot = []
epoch_plot = []
param_plot = []
c_list = None

num_qubits = 7
parameters = np.random.normal(0, 2 * np.pi, 1000)
c = CircuitsWords('C:/Users/usuario/Desktop/QIT/QNNs/qnn/qnlp/data/DataBase_docs - Hoja 1.csv', num_qubits, 2)

circuits = c.create()
print(circuits)
results = c.sample_run_global(parameters, 100)
expected_bits = get_expected_bits(c.df, 2, num_qubits)
print(expected_bits)
result = get_overall_run_words(results[1], 1)
for i in results:
	print(i)

print(c.params_used)

lr = 1
epsilon = .0000001

for o in range(50):
	results = c.sample_run_global(parameters, 100)
	cost_ = cost_global(results, expected_bits)
	cost_plot.append(cost_)
	param_plot.append(parameters)
	epoch_plot.append(o)

	for i in c.params_used:
		delta = g_parameter_shift_global_words(c, i, parameters, expected_bits)
		print(i, 'done!')
		parameters[i] = parameters[i] - lr * (delta + epsilon**2)
	print('Epoch:', o, 'Cost:', cost_plot[o], 'LearningRate:', lr)

min_i = 0
min_cost = cost_plot[0]
for i in range(len(cost_plot)):
	if cost_plot[i] < min_cost:
		min_cost = cost_plot[i]
		min_i = i

print('min COST:', min_cost)
print('min EPOCH:', min_i)
print('min PARAM:', param_plot[min_i])

plt.plot(epoch_plot, cost_plot)
plt.show()
print(param_plot[min_i])

import numpy as np

from qnn.qnlp.circuits_words import CircuitsWords
from qnn.qnlp.optimization import cost_global
from qnn.qnlp.optimization_words import g_parameter_shift_global_words
from qnn.qnlp.optimization_words import get_overall_run_words

parameters = np.random.normal(0, 2 * np.pi, 1000)
c = CircuitsWords('C:/Users/usuario/Desktop/QIT/QNNs/qnn/qnlp/data/3Q DataBase.csv')

circuits = c.create()

results = c.sample_run_global(parameters, 100)
for i in results:
	print(i)

result = get_overall_run_words(results[1], 1)

lr = .1
epsilon = .0000001

for o in range(10):
	print(cost_global(results, [[0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 0, 0]]))
	for i in range(len(parameters)):
		parameters[i] = parameters[i] - lr * (g_parameter_shift_global_words(c, i, parameters, [[0, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 0, 0]]) + epsilon**2)
		print('done', i)
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from qnn.qnlp.circuits_words import CircuitsWords
from qnn.qnlp.numbers.optimization import cost_global
from qnn.qnlp.optimization_words import g_parameter_shift_global_words, get_expected_bits

cost_plot, epoch_plot, param_plot = [], [], []  # lists for plotting the results at the end
c_list = None

num_qubits = 7  # 7 qubits to use on the model
num_iterations = 4  # iterations of the optimization (the number of times that the parameters are updated)
parameters = np.random.normal(0, 2 * np.pi, 1000)  # initialize random parameters for the gates between 0 and 2π
# TODO: specify the path of the csv file used as database
database_path = 'C:/Users/usuario/Desktop/QIT/QNNs/qnn/qnlp/data/DataBase_docs - Hoja 1.csv'

# creates the CircuitsWords class for the 6 firsts phrases on the csv file
# (the creation of this class only creates the gates associated with each word used in the phrases)
c = CircuitsWords(database_path, num_qubits, 6)

circuits = c.create()  # creates the cirq.Circuit() from the CircuitsWords class
print(circuits)
results = c.sample_run_global(parameters, 100)  # simulates and evaluates the circuits
expected_bits = get_expected_bits(c.df, 6, num_qubits)
print(c.words_used)
print(expected_bits)
print(c.params_used)

lr, epsilon = 1, .0000001  # Defines the values used for the calculation of the gradient
print('\n \n \n', 'Start:', datetime.now().time())  # prints the time when the optimization started

# goes thought the iterations of the optimization
for o in range(num_iterations):
	results = c.sample_run_global(parameters, 100)
	cost_ = cost_global(results, expected_bits)  # evaluate the cost
	# and appends cost_ to the cost_plot list for the plotting
	# as well with the parameters and the humber of the epoch (iteration)
	cost_plot.append(cost_)
	param_plot.append(parameters)
	epoch_plot.append(o)

	# goes thought all the parameters that are used by the gates of the words used at the sentences
	# and updates them according with the gradient to respect to the cost function
	for i in c.params_used:
		delta = g_parameter_shift_global_words(c, i, parameters, expected_bits)
		parameters[i] = parameters[i] - lr * (delta + epsilon**2)
	# print the info of the iteration
	print('Epoch:', o, 'Cost:', cost_plot[o], 'LearningRate:', lr, datetime.now().time())

# finds the parameters that have the minimal cost and use them to display the final results
min_i = 0
min_cost = cost_plot[0]
for i in range(len(cost_plot)):
	if cost_plot[i] < min_cost:
		min_cost = cost_plot[i]
		min_i = i

print('min COST:', min_cost)
print('min EPOCH:', min_i)
print('min PARAM:', param_plot[min_i])

# prints the raw results of the circuits to see how close are to the expected bits
result = c.sample_run_global(param_plot[min_i], 100)
for i in result:
	print(i, '\n')

# plots the graph of the cost with respect to the iterations
plt.plot(epoch_plot, cost_plot)
plt.show()

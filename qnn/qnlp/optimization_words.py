import cirq
import numpy as np
import pandas as pd
from typing import List

from qnn.qnlp.circuits_words import CircuitsWords


def get_overall_run_words(trial_result: cirq.TrialResult, num: int):
	""" Takes the average of the measurements of a given qubit on a given circuit
	(the results are on the form of a bitstring)
	If the qubit is in state $|1>$ all the measurement are going to be 1 forming a bitstring of then length of
	number of measurement and consisting of all ones.

	Args:
	    trial_result (cirq.TrialResult): the result of the simulation of a specific circuit
	    num (int): the number of the qubit on which the average is taken (starting from 0)

	Returns:
	    float: a value between 1 and 0 that is the average of the measurements of the specified qubit
	"""
	a = 0
	index_num = 0
	string = '(' + str(num) + ', 0)'  # localizes the qubit specified by num
	# (in the cirq.TrialResult object are in the form "(x, 0):", with x as the number of the qubit)
	for i in trial_result.data.columns:  # using the argument data on cirq.TrialResult is transformed to a pd.DataFrame
		if i == string:
			index_num = a
		a = a + 1
	dict_result = trial_result.data.transpose()
	values = dict_result.values[index_num]  # gets the bitstring of the specified qubit
	# values is a numpy array
	return sum(values) / len(values)  # sum the bits and normalizes them according to the len of the bitstring


def cost_global_words(trial_results: List[cirq.TrialResult], expected_bits: List[List[float]]):
	""" Given a list of circuits results, returns the global cost of those results doing a average of the local cost
	of each circuits. The local cost evaluated using the  mean squared error method. And the global cost is the average
	of all the local cost of all the circuits (that are used in the optimization).
	The cost would be zero in the case that all qubits are in the desired basis state (|0> or |1>).

	Args:
		trial_results: a list of cirq.TrialResult of the circuits on which the cost is taken
		expected_bits: a list of the bits that the qubits are supposed to be at (0 for qubit in |0> and 1 for |1>)
		(e.g. bits = [[1, 0, 0], [0, 1, 1]] where bits[0] are the expected bits for the first circuit)

	Returns:
		float: the global cost evaluated using mean squared error
	"""

	result, result_global = [], []
	for a in range(len(trial_results)):
		for i in range(len(trial_results[a].data.columns)):  # get the result of a specific qubit
			# calculate the mean squared error   (Y' - Y)^2
			result.append((get_overall_run_words(trial_results[a], i) - (expected_bits[a])[i])**2)
		# adjust the sum of the results to the number of measurements
		result_global.append((1 / 2 * len(trial_results[a].data.columns)) * sum(result))
	# evaluate the average of all the local cost
	return sum(result_global) / len(trial_results)


def g_parameter_shift_global_words(circuits_object: CircuitsWords,
                                   param: int,
                                   theta_sample: np.array,
                                   expected_bits: List[List[float]]):
	""" Given a CircuitsWords and a parameter, takes the gradient of the circuits (all of the circuits in the object)
witch respect to that parameter.
The parameter shift method is the one used (Eq. 2)
	Args:
		circuits_object: the circuit on which the gradient is taken
		param: the parameter that the gradient is taken respected to
		theta_sample: the parameters of the circuit
		expected_bits: a list of the bits that the qubits are supposed to be at (to evaluate the cost function)

	Returns:
		float: The gradient of the circuits with respect to a specific parameter
	"""
	# creates the perturbation vector
	perturbation_vector = np.zeros(len(theta_sample))
	perturbation_vector[param] = 1

	# creates the new parameters (only updating the parameter specified with param)
	pos_theta = theta_sample + (np.pi / 4) * perturbation_vector
	neg_theta = theta_sample - (np.pi / 4) * perturbation_vector

	# simulates the new results with CircuitsWords.sample_run_global()
	pos_result = circuits_object.sample_run_global(pos_theta, 100)
	neg_result = circuits_object.sample_run_global(neg_theta, 100)

	return cost_global_words(pos_result, expected_bits) - cost_global_words(neg_result, expected_bits)


def get_expected_bits(data_frame: pd.DataFrame,
                      num_phrases: int,
                      num_qubits: int):
	""" Gets the expected bits of the circuits according to they desired output.
The number of phrases and the number of qubits is needed, that is in case the optimization is on fewer phrases that the
ones on the database. The number of qubits is needed because the length of the expected bits has to match with it.

	Args:
		data_frame: the DataFrame that corresponds to the database (using the function extract_words)
		num_phrases: the number of phrases used in the optimization
		num_qubits: the number of qubits used in the optimization

	Returns:
		List: expected bits of all the circuits
	"""
	expected_bits = [[] for __ in range(num_phrases)]
	a = 0
	for j in data_frame.transpose().values[4][:num_phrases]:
		index = 0
		for i in data_frame.transpose().values[0]:
			if i == j:
				break
			index += 1
			if index == 84:
				raise ValueError('Word not found')

		bit = bin(index)[2:]
		for _ in range(num_qubits - len(bit)):
			expected_bits[a].append(0)
		for __ in bit:
			expected_bits[a].append(int(__))
		a += 1
	return expected_bits

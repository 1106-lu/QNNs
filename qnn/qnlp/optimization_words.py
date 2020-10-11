import cirq
import numpy as np
import pandas as pd
from typing import List

from qnn.qnlp.circuits_words import CircuitsWords


def get_overall_run_words(trial_result: cirq.TrialResult, num):
	a = 0
	index_num = 0
	string = '(' + str(num) + ', 0)'
	for i in trial_result.data.columns:
		if i == string:
			index_num = a
		a = a + 1
	dict_result = trial_result.data.transpose()
	values = dict_result.values[index_num]
	return sum(values) / len(values)


def cost_global_words(trial_results: List[cirq.TrialResult], expected_bits: List[List[float]]):
	result, result_global = [], []

	for a in range(len(trial_results)):
		for i in range(len(trial_results[a].data.columns)):
			result.append((get_overall_run_words(trial_results[a], i) - (expected_bits[a])[i])**2)
		result_global.append((1 / 2 * len(trial_results[a].data.columns)) * sum(result))

	return sum(result_global) / len(trial_results)


def g_parameter_shift_global_words(circuits_object: CircuitsWords,
                                   param,
                                   theta_sample,
                                   expected_bits: List[List[float]]):
	perturbation_vector = np.zeros(len(theta_sample))
	perturbation_vector[param] = 1

	pos_theta = theta_sample + (np.pi / 4) * perturbation_vector
	neg_theta = theta_sample - (np.pi / 4) * perturbation_vector

	pos_result = circuits_object.sample_run_global(pos_theta, 1000)
	neg_result = circuits_object.sample_run_global(neg_theta, 1000)

	return cost_global_words(pos_result, expected_bits) - cost_global_words(neg_result, expected_bits)


def get_expected_bits(data_frame: pd.DataFrame, num_phrases: int):
	expected_bits = [[] for __ in range(num_phrases)]
	a = 0
	for j in data_frame.transpose().values[4][:21]:
		index = 0
		for i in data_frame.transpose().values[0]:
			if i == j:
				break
			index += 1
		bit = bin(index)[2:]
		for _ in range(7 - len(bit)):
			expected_bits[a].append(0)
		for l in bit:
			expected_bits[a].append(int(l))
		a += 1
	return expected_bits

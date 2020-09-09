import cirq
import numpy as np
from typing import List

from qnn.qnlp.circuits_numbers import sample_run, sample_run_global


def get_overall_run(trial_result: cirq.TrialResult, num):
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


def cost_1(trial_result: cirq.TrialResult, word_num):
	not_goal = []
	for i in range(len(trial_result.data.columns)):
		if i != word_num:
			not_goal.append(get_overall_run(trial_result, i))
	result = get_overall_run(trial_result, word_num) - (sum(not_goal) / 4)
	return result**2


def cost_2(trial_result: cirq.TrialResult, expected_bits):
	result = []
	for i in range(len(trial_result.data.columns)):
		result.append((get_overall_run(trial_result, i) - expected_bits[i])**2)
	return (1 / 2 * len(trial_result.data.columns)) * sum(result)


def cost_global(trial_results: List[cirq.TrialResult], expected_bits: List[List[float]]):
	result, result_global = [], []

	for a in range(len(trial_results)):
		for i in range(len(trial_results[a].data.columns)):
			result.append((get_overall_run(trial_results[a], i) - (expected_bits[a])[i])**2)
		result_global.append((1 / 2 * len(trial_results[a].data.columns)) * sum(result))

	return sum(result_global) / len(trial_results)


def g_finite_difference(circuits, param, theta_sample, epsilon: float, expected_bits):
	perturbation_vector = np.zeros(len(theta_sample))
	perturbation_vector[param] = 1

	pos_theta = theta_sample + epsilon * perturbation_vector
	neg_theta = theta_sample - epsilon * perturbation_vector

	pos_result = sample_run(circuits, pos_theta, 1000)
	neg_result = sample_run(circuits, neg_theta, 1000)

	result = (cost_2(pos_result, expected_bits) - cost_2(neg_result, expected_bits)) / 2 * epsilon

	return result


def g_parameter_shift(circuits, param, theta_sample, expected_bits):
	perturbation_vector = np.zeros(len(theta_sample))
	perturbation_vector[param] = 1
	neg_theta = theta_sample - (np.pi / 4) * perturbation_vector
	pos_theta = theta_sample + (np.pi / 4) * perturbation_vector

	neg_vec = sample_run(circuits, neg_theta, 1000)
	pos_vec = sample_run(circuits, pos_theta, 1000)

	result = cost_2(pos_vec, expected_bits) - cost_2(neg_vec, expected_bits)
	return result


def g_parameter_shift_global(circuits: List[cirq.Circuit],
                             param,
                             theta_sample,
                             expected_bits: List[List[float]]):
	perturbation_vector = np.zeros(len(theta_sample))
	perturbation_vector[param] = 1

	pos_theta = theta_sample + (np.pi / 4) * perturbation_vector
	neg_theta = theta_sample - (np.pi / 4) * perturbation_vector

	pos_result = sample_run_global(circuits, pos_theta, 1000)
	neg_result = sample_run_global(circuits, neg_theta, 1000)

	return cost_global(pos_result, expected_bits) - cost_global(neg_result, expected_bits)

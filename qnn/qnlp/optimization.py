from typing import Any, Union

import numpy as np
import cirq
from qnn.qnlp.circuits_numbers import sample_run


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


def cost(trial_result: cirq.TrialResult, word_num):
	not_goal = []

	for i in range(len(trial_result.data.columns)):
		if i != word_num:
			not_goal.append(get_overall_run(trial_result, i))

	result = get_overall_run(trial_result, word_num) - sum(not_goal)**2
	return result


def g_finite_difference(circuits, param, theta_sample, epsilon):
	perturbation_vector = np.zeros(len(theta_sample))
	perturbation_vector[param] = 1

	neg_theta = theta_sample - epsilon * perturbation_vector
	pos_theta = theta_sample + epsilon * perturbation_vector

	neg_result = sample_run(circuits, neg_theta, 100)
	pos_result = sample_run(circuits, pos_theta, 100)

	result = (cost(pos_result, 1) - cost(neg_result, 1) / 2 * epsilon)
	return result

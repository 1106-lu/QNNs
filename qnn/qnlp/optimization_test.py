from qnn.qnlp.circuits_numbers import create_circuits, sample_run
from qnn.qnlp.optimization import get_overall_run, cost
import numpy as np
import cirq

parameters = np.random.normal(0, 2*np.pi, 33)

c = create_circuits()

print(c)

result_run = sample_run(c, theta_sample=parameters, repetitions=10)
print(result_run)
print(result_run.data.transpose().values)

print(cost(result_run, 1))

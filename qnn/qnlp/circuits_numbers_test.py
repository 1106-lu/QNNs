from qnn.qnlp.circuits_numbers import create_circuits, sample_run, sample_simulate
import numpy as np

parameters = np.random.normal(0, 2*np.pi, 33)

c = create_circuits()

result_simulate = sample_simulate(c, theta_sample=parameters)
print('\n Simulations results: \n ', result_simulate, '\n \n')

result_run = sample_run(c, theta_sample=parameters, repetitions=100)
print('\n Run results: \n', result_run, '\n \n')

print(c)

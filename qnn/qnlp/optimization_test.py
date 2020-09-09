import cirq
import numpy as np

from qnn.qnlp.circuits_numbers import sample_run, cc_12_34, sample_run_global
from qnn.qnlp.optimization import cost_2, cost_global

q = []
parameters = np.random.normal(0, 2*np.pi, 20)
for i in range(5):
	q.append(cirq.GridQubit(i, 0))

c = cc_12_34()
result_global = []

for i in c:
	print(i)
	result_run = sample_run(i, theta_sample=parameters, repetitions=100)
	print(result_run)
	#  print(result_run.data.transpose().values)

	print('With cost_2', cost_2(result_run, [0, 1, 0, 0, 0]))

result_global = sample_run_global(c, theta_sample=parameters, repetitions=100)
print(cost_global(result_global, [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]))

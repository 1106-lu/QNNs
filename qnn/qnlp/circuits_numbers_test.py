import numpy as np

from qnn.qnlp.circuits_numbers import cc_1234567, sample_run_global

parameters = np.random.normal(0, 2*np.pi, 32)

c = cc_1234567()
for i in c:
	print(i)

results = sample_run_global(c, parameters, 100)
for i in results:
	print(i)